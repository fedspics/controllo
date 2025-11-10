#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import tf2_ros
import math
import numpy as np
import time
import traceback
import cv2


class NodoDeteccionCaidas(Node):
    def __init__(self):
        super().__init__('pose_detection_node')

        # --- QoS ---
        qos_camera = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Nav2 ---
        self.navigator = BasicNavigator()
        self.get_logger().info('Attendo Nav2...')
        try:
            self.navigator.waitUntilNav2Active(timeout_sec=15.0)
            self.get_logger().info('Nav2 attivo ✅')
        except Exception:
            self.get_logger().warn('Nav2 non attivo / timeout, continuo comunque.')

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- YOLO / CvBridge ---
        self.bridge = CvBridge()
        self.modelo = YOLO('yolov8n-pose.pt')

        # --- Publishers / Subscribers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rgb_sub = self.create_subscription(Image, '/head_front_camera/rgb/image_raw', self.rgb_callback, qos_camera)
        self.depth_sub = self.create_subscription(Image, '/head_front_camera/depth/image_raw', self.depth_callback, qos_camera)
        self.info_sub = self.create_subscription(CameraInfo, '/head_front_camera/depth/camera_info', self.info_callback, qos_camera)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_map)

        # --- Buffers sensori ---
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_info = None
        self.latest_map = None

        # --- Flags e stato MSF ---
        self.estado = 'inicial'
        self.alerta_enviada = False

        # --- Parametri ---
        self.inferences_per_phase = 10
        self.rotation_steps = 8
        self.rotation_speed = 0.5
        self.yaw_tolerance = 0.04
        self.wp_step = 1.5
        self.navigator_lock = False

        # --- Variabili interne ---
        self._infer_count = 0
        self._rot_step_idx = 0
        self._rot_target_yaw = None
        self._rotating_phase = 'idle'
        self._inf_count_at_stop = 0
        self.waypoints = []
        self.waypoint_index = 0
        self._nav_goal_sent = False
        self._nav_timeout = 60.0
        self._nav_start_time = None
        self._stopping = False
        self._initial_scan_done = False

        # --- Mappa esplorazione ---
        self.visited_cells = set()
        self.cell_size = 1.0  # m
        self.safe_clearance_m = 0.5  # distanza minima da ostacoli

        # --- Timer ---
        self.timer = self.create_timer(0.08, self.main_loop)

        self.get_logger().info('NodoDeteccionCaidas avviato con logica di esplorazione migliorata.')

    # ---------- Callbacks ----------
    def rgb_callback(self, msg): self.latest_rgb = msg
    def depth_callback(self, msg): self.latest_depth = msg
    def info_callback(self, msg): self.latest_info = msg
    def map_callback(self, msg): self.latest_map = msg

    # ---------- Inferenza singola ----------
    def infer_one(self):
        if self.latest_rgb is None:
            return None, None, None, None
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding='bgr8')
        except Exception:
            return None, None, None, None

        depth_img = None
        info_msg = None
        if self.latest_depth is not None and self.latest_info is not None:
            try:
                depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
                info_msg = self.latest_info
            except Exception:
                pass

        try:
            resultados = self.modelo(img_bgr, conf=0.3)
            if len(resultados) == 0:
                return None, img_bgr, depth_img, info_msg
            return resultados[0], img_bgr, depth_img, info_msg
        except Exception:
            self.get_logger().error('Errore infer_one:\n' + traceback.format_exc())
            return None, img_bgr, depth_img, info_msg

    # ---------- Keypoints e caduta ----------
    def keypoints_visibles(self, keypoints):
        try:
            cabeza_visible = any(keypoints[i][2] > 0.5 for i in [0, 1, 2])
            pies_visible = any(keypoints[i][2] > 0.5 for i in [15, 16])
            return cabeza_visible and pies_visible
        except Exception:
            return False

    def comprobar_caida_3d(self, keypoints, depth_img, info_msg):
        if info_msg is None or depth_img is None:
            return False
        try:
            K = np.array(info_msg.k).reshape(3, 3)
        except Exception:
            return False

        def pixel_a_3d(u, v, depth):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            return np.array([x, y, z])

        def punti_media(indices):
            pts = []
            for i in indices:
                u, v, conf = keypoints[i]
                if conf > 0.5:
                    ui, vi = int(u), int(v)
                    if 0 <= vi < depth_img.shape[0] and 0 <= ui < depth_img.shape[1]:
                        d = float(depth_img[vi, ui])
                        if 0.05 < d < 10.0:
                            pts.append(pixel_a_3d(ui, vi, d))
            return np.mean(pts, axis=0) if pts else None

        cabeza = punti_media([0, 1, 2])
        pies = punti_media([15, 16])
        if cabeza is None or pies is None:
            return False

        dist_vertical = abs(cabeza[1] - pies[1])
        dist_3d = np.linalg.norm(cabeza - pies)
        ratio = dist_vertical / dist_3d if dist_3d > 0 else 0

        if dist_3d < 1.2:
            soglia_verticale = 0.35
            soglia_ratio = 0.5
        else:
            soglia_verticale = 0.6
            soglia_ratio = 0.7

        return (dist_vertical < soglia_verticale) and (ratio < soglia_ratio)

    # ---------- Stop sicuro ----------
    def safe_publish_stop(self):
        if rclpy.ok():
            try:
                self.cmd_vel_pub.publish(Twist())
            except Exception:
                pass

    # ---------- Waypoints dinamici intelligenti ----------
    def genera_waypoints_dinamici(self):
        if self.latest_map is None:
            self.get_logger().warn("Mappa non disponibile: generazione locale.")
            return []

        map_data = np.array(self.latest_map.data, dtype=np.int8)
        width = self.latest_map.info.width
        height = self.latest_map.info.height
        resolution = self.latest_map.info.resolution
        origin = self.latest_map.info.origin.position

        map_grid = map_data.reshape(height, width)

        # crea maschera libera
        free_mask = (map_grid == 0)
        # dilata gli ostacoli per creare zona di sicurezza
        kernel = np.ones((int(self.safe_clearance_m / resolution), int(self.safe_clearance_m / resolution)), np.uint8)
        obstacle_mask = cv2.dilate((map_grid > 0).astype(np.uint8), kernel)
        safe_mask = np.logical_and(free_mask, obstacle_mask == 0)

        try:
            pose = self.navigator.getCurrentPose().pose.position
            robot_x, robot_y = pose.x, pose.y
        except Exception:
            robot_x, robot_y = 0.0, 0.0

        ix = int((robot_x - origin.x) / resolution)
        iy = int((robot_y - origin.y) / resolution)

        waypoints = []
        step_px = int(self.cell_size / resolution)
        search_radius = int(5.0 / resolution)

        for dx in range(-search_radius, search_radius, step_px):
            for dy in range(-search_radius, search_radius, step_px):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if safe_mask[ny, nx]:
                        cell_id = (nx // step_px, ny // step_px)
                        if cell_id not in self.visited_cells:
                            self.visited_cells.add(cell_id)
                            wx = origin.x + nx * resolution
                            wy = origin.y + ny * resolution
                            goal = PoseStamped()
                            goal.header.frame_id = 'map'
                            goal.header.stamp = self.get_clock().now().to_msg()
                            goal.pose.position.x = float(wx)
                            goal.pose.position.y = float(wy)
                            goal.pose.orientation = Quaternion(w=1.0)
                            waypoints.append(goal)

        if not waypoints:
            self.get_logger().warn("Tutti i waypoint vicini esplorati, reset esplorazione.")
            self.visited_cells.clear()
            return self.genera_waypoints_dinamici()

        self.get_logger().info(f"Generati {len(waypoints)} waypoint liberi e distanti dagli ostacoli.")
        return waypoints

    # ---------- Utility ----------
    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalizza_angolo(self, ang):
        while ang > math.pi:
            ang -= 2 * math.pi
        while ang < -math.pi:
            ang += 2 * math.pi
        return ang

    # ---------- MAIN ----------
    def main_loop(self):
        if self._stopping or not rclpy.ok():
            return
        try:
            # ----------------- STATO INICIAL -----------------
            if self.estado == 'inicial':
                if not self._initial_scan_done:
                    if self._infer_count == 0:
                        self.get_logger().info("Prima scansione iniziale (10 inferenze).")
                    if self._infer_count < self.inferences_per_phase:
                        res, img, depth_img, info_msg = self.infer_one()
                        self._infer_count += 1
                        if res is not None and hasattr(res, 'keypoints') and res.keypoints is not None and res.keypoints.has_visible:
                            try:
                                kp = res.keypoints[0].cpu().numpy().data
                                if self.keypoints_visibles(kp):
                                    self.get_logger().info("Persona rilevata durante scansione iniziale.")
                                    self._infer_count = 0
                                    return
                                if self.comprobar_caida_3d(kp, depth_img, info_msg):
                                    if not self.alerta_enviada:
                                        self.alerta_enviada = True
                                        self.get_logger().warn("⚠️ CADUTA rilevata in INICIAL!")
                                        self.safe_publish_stop()
                                        self.navigator.cancelTask()
                                    self._infer_count = 0
                                    return
                            except Exception:
                                self.get_logger().error("Errore keypoints iniziale:\n" + traceback.format_exc())
                        return
                    else:
                        self._infer_count = 0
                        self._initial_scan_done = True
                        self.get_logger().info("Scansione iniziale completata → passo a ROTANDO.")
                        self.estado = 'rotando'
                        self._rot_step_idx = 0
                        self._rotating_phase = 'turning'
                        try:
                            tf_init = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
                            self._rot_base_yaw = self.quaternion_to_yaw(tf_init.transform.rotation)
                        except Exception:
                            self._rot_base_yaw = 0.0
                        return
                else:
                    # già fatta, passa direttamente a rotando
                    self.estado = 'rotando'
                    return

            # ----------------- STATO ROTANDO -----------------
            if self.estado == 'rotando':
                if self._rot_step_idx >= self.rotation_steps:
                    self.get_logger().info("Rotazione completa: nessuna persona trovata.")
                    if not self.waypoints:
                        self.waypoints = self.genera_waypoints_dinamici()
                        self.waypoint_index = 0
                    if self.waypoints:
                        self._nav_goal_sent = False
                        self.estado = 'navegando'
                        self._nav_start_time = None
                    else:
                        self.estado = 'inicial'
                    return

                if self._rotating_phase == 'turning':
                    if self._rot_target_yaw is None:
                        step = self._rot_step_idx + 1
                        self._rot_target_yaw = self.normalizza_angolo(
                            self._rot_base_yaw + step * (2 * math.pi / self.rotation_steps)
                        )
                    try:
                        tf_now = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
                        yaw_now = self.quaternion_to_yaw(tf_now.transform.rotation)
                    except Exception:
                        return
                    errore = self.normalizza_angolo(self._rot_target_yaw - yaw_now)
                    if abs(errore) < self.yaw_tolerance:
                        self.safe_publish_stop()
                        self._rotating_phase = 'inferring'
                        self._inf_count_at_stop = 0
                        return
                    twist = Twist()
                    twist.angular.z = self.rotation_speed if errore > 0 else -self.rotation_speed
                    self.cmd_vel_pub.publish(twist)
                    return

                if self._rotating_phase == 'inferring':
                    if self._inf_count_at_stop < self.inferences_per_phase:
                        res, img, depth_img, info_msg = self.infer_one()
                        self._inf_count_at_stop += 1
                        if res is not None and hasattr(res, 'keypoints') and res.keypoints is not None and res.keypoints.has_visible:
                            try:
                                kp = res.keypoints[0].cpu().numpy().data
                                if self.keypoints_visibles(kp):
                                    self.get_logger().info("Persona trovata durante rotazione → torno a INICIAL.")
                                    self.estado = 'inicial'
                                    self._rotating_phase = 'idle'
                                    self._rot_target_yaw = None
                                    self._rot_step_idx = 0
                                    return
                                if self.comprobar_caida_3d(kp, depth_img, info_msg):
                                    if not self.alerta_enviada:
                                        self.alerta_enviada = True
                                        self.get_logger().warn("⚠️ CADUTA rilevata durante rotazione!")
                                        self.safe_publish_stop()
                                        self.navigator.cancelTask()
                                    self.estado = 'inicial'
                                    return
                            except Exception:
                                self.get_logger().error('Errore parsing keypoints durante rotazione.')
                        return
                    else:
                        self._rot_step_idx += 1
                        self._rot_target_yaw = None
                        self._rotating_phase = 'turning'
                        return

            # ----------------- STATO NAVEGANDO -----------------
            if self.estado == 'navegando':
                if not self._nav_goal_sent:
                    if self.waypoint_index >= len(self.waypoints):
                        self.estado = 'inicial'
                        self.waypoint_index = 0
                        return
                    goal_pose = self.waypoints[self.waypoint_index]
                    try:
                        self.navigator.goToPose(goal_pose)
                        self._nav_goal_sent = True
                        self._nav_start_time = time.time()
                    except Exception:
                        self.get_logger().error('Errore goToPose.')
                        self.estado = 'inicial'
                    return

                try:
                    if self.navigator.isTaskComplete():
                        result = self.navigator.getResult()
                        if result == TaskResult.SUCCEEDED:
                            self.get_logger().info("Arrivato al waypoint.")
                        self.waypoint_index += 1
                        self._nav_goal_sent = False
                        self.estado = 'rotando'
                        return
                except Exception:
                    self.navigator.cancelTask()
                    self.estado = 'inicial'
                    return

                if self._nav_start_time and (time.time() - self._nav_start_time) > self._nav_timeout:
                    self.get_logger().warn("Timeout navigazione.")
                    self.navigator.cancelTask()
                    self._nav_goal_sent = False
                    self.estado = 'rotando'
                    return

        except Exception:
            self.get_logger().error('Errore main_loop:\n' + traceback.format_exc())
            self.safe_publish_stop()
            self.estado = 'inicial'
            self._rotating_phase = 'idle'
            self._rot_target_yaw = None
            self._rot_step_idx = 0


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoDeteccionCaidas()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        nodo.get_logger().info("Chiusura nodo.")
    finally:
        nodo.safe_publish_stop()
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

