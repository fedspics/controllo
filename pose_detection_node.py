import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion
from cv_bridge import CvBridge
from ultralytics import YOLO
import tf2_ros
import tf_transformations
import math
import time
from nav2_simple_commander.robot_navigator import BasicNavigator

class PoseDetectionNode(Node):
    def __init__(self):
        super().__init__('pose_detection_node')

        self.navigator = BasicNavigator()
        self.get_logger().info('Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Nav2 is ready for use!')

        self.subscription = self.create_subscription(
            Image,
            '/head_front_camera/rgb/image_raw',
            self.image_callback,
            10)

        self.bridge = CvBridge()
        self.model = YOLO('yolov8n-pose.pt')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.moving = False
        self.alert_sent = False
        self.last_known_person_pose = None
        self.frames_buffer = []
        self.min_valid_detections = 3
        self.latest_image_msg = None

    def image_callback(self, msg):
        if self.moving:
            return

        self.latest_image_msg = msg
        self.get_logger().info("Ricevuto frame immagine, eseguo inferenza...")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image)

        if not results or len(results[0].keypoints) == 0:
            self.frames_buffer.append("no_detection")
        else:
            keypoints = results[0].keypoints[0].cpu().numpy()
            if keypoints.shape[0] >= 17:
                head_visible = any(keypoints[i][2] > 0.3 for i in [0, 1])
                feet_visible = any(keypoints[i][2] > 0.3 for i in [15, 16])
                if head_visible and feet_visible:
                    self.frames_buffer.append("ok")
                    if len([f for f in self.frames_buffer[-self.min_valid_detections:] if f == "ok"]) >= self.min_valid_detections:
                        self.process_valid_detection(keypoints)
                    return
                else:

                    pose = self.get_robot_pose()
                    if pose:
                        self.last_known_person_pose = pose
                    self.frames_buffer.append("incomplete")
            else:
                self.frames_buffer.append("incomplete")

        if self.should_trigger_search():
            self.get_logger().warn("Troppe rilevazioni negative o incomplete. Avvio strategie.")
            if "incomplete" in self.frames_buffer[-self.min_valid_detections:] and self.last_known_person_pose:
                self.try_back_and_face()
            else:
                self.search_person()

    def should_trigger_search(self):
        if len(self.frames_buffer) < self.min_valid_detections:
            return False
        recent = self.frames_buffer[-self.min_valid_detections:]
        return all(f != "ok" for f in recent)

    def process_valid_detection(self, keypoints):
        self.get_logger().info("Keypoints completi rilevati.")
        if self.alert_sent:
            self.get_logger().info("Persona si è rialzata, resetto avviso caduta.")
            self.alert_sent = False

        fallen = self.check_fall(keypoints)
        if fallen and not self.alert_sent:
            self.get_logger().warn("Caduta rilevata!")
            self.alert_sent = True

        pose = self.get_robot_pose()
        if pose:
            self.last_known_person_pose = pose

    def check_fall(self, keypoints):
        head_y = keypoints[0][1]
        foot_y = max(keypoints[15][1], keypoints[16][1])
        return head_y > foot_y - 20

    def get_robot_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform('map', 'base_link', now, timeout=rclpy.duration.Duration(seconds=1.0))
            return transform.transform.translation
        except Exception as e:
            self.get_logger().warn(f"Errore nel recupero trasformazione: {e}")
            return None

    def move_to(self, x, y, yaw=0.0):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)
        goal.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        self.moving = True
        self.get_logger().info(f"Navigo a: {x:.2f}, {y:.2f} con orientamento {yaw:.2f} rad.")
        self.navigator.goToPose(goal)

        start_time = time.time()
        while not self.navigator.isTaskComplete():
            if time.time() - start_time > 20:
                self.get_logger().error("Timeout nel movimento.")
                self.navigator.cancelTask()
                self.moving = False
                return False
            time.sleep(0.5)

        self.get_logger().info("Movimento completato.")
        self.moving = False
        return True

    def rotate_and_scan(self):
        for angle_deg in range(0, 360, 45):
            yaw = math.radians(angle_deg)
            pose = self.get_robot_pose()
            if pose:
                self.move_to(pose.x, pose.y, yaw)
                time.sleep(1.5)
                if self.latest_image_msg:
                    self.image_callback(self.latest_image_msg)

    def try_back_and_face(self):
        pose = self.get_robot_pose()
        target = self.last_known_person_pose
        if pose and target:
            dx = pose.x - target.x
            dy = pose.y - target.y
            norm = math.hypot(dx, dy)
            dx /= (norm + 1e-6)
            dy /= (norm + 1e-6)
            back_x = pose.x + dx * 0.7
            back_y = pose.y + dy * 0.7
            self.get_logger().info("Indietreggio per migliorare l'inquadratura.")
            self.move_to(back_x, back_y)
            yaw = math.atan2(target.y - back_y, target.x - back_x)
            self.move_to(back_x, back_y, yaw)

    def search_person(self):
        self.get_logger().info("Inizio ricerca persona nella mappa...")
        base_pose = self.get_robot_pose()
        if not base_pose:
            return

        waypoints = []
        for dx in [-1.0, 0.0, 1.0]:
            for dy in [-1.0, 0.0, 1.0]:
                if dx == 0 and dy == 0:
                    continue
                x = base_pose.x + dx
                y = base_pose.y + dy
                waypoints.append((x, y))

        for (x, y) in waypoints:
            if self.move_to(x, y):
                self.rotate_and_scan()

def main(args=None):
    rclpy.init(args=args)
    node = PoseDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

