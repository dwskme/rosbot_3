import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import time
from rosbot_lane_follower_v2.unet_model import UNet
from rosbot_lane_follower_v2.transforms import get_transforms
from rosbot_lane_follower_v2.utils import find_lane_center

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_pub = self.create_publisher(Image, "/camera/color/debug", 10)

        model_path = self.declare_parameter(
            "model_path", 
            "/home/wheeltec/ros_ws3/src/rosbot_lane_follower_v2/models/lane_unet.pth"
        ).get_parameter_value().string_value

        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transforms = get_transforms()
        self.subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            10
        )
        self.center_pub = self.create_publisher(Float32, "/lane_center", 10)

        self.last_time = time.time()
        self.last_cam_time = time.time()
        self.fps_log = []
        self.cam_fps_log = []
        self.frame_count = 0

    def image_callback(self, msg):
        # === FPS tracking ===
        self.frame_count += 1
        now = time.time()

        # Inference FPS
        inf_fps = 1.0 / (now - self.last_time)
        self.fps_log.append(inf_fps)
        if len(self.fps_log) > 30:
            self.fps_log.pop(0)
        avg_inf_fps = sum(self.fps_log) / len(self.fps_log)

        # Camera FPS
        cam_fps = 1.0 / (now - self.last_cam_time)
        self.cam_fps_log.append(cam_fps)
        if len(self.cam_fps_log) > 30:
            self.cam_fps_log.pop(0)
        avg_cam_fps = sum(self.cam_fps_log) / len(self.cam_fps_log)
        self.last_cam_time = now

        self.get_logger().info(f"Inference FPS: {avg_inf_fps:.2f} | Camera FPS: {avg_cam_fps:.2f}")
        self.last_time = now

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = img.shape[:2]
        img = img[:, 20:]  # offset camera fix: crop 20px from left

        # Resize and transform
        input_img = cv2.resize(img, (256, 256))
        input_tensor = self.transforms(image=input_img)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()

        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Calculate lane center
        center_x = find_lane_center(mask)
        self.center_pub.publish(Float32(data=center_x))

        # === Overlay FPS info on original image ===
        overlay_img = img.copy()
        cv2.putText(overlay_img, f"Inf FPS: {avg_inf_fps:.1f} | Cam FPS: {avg_cam_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
