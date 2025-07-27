#!/usr/bin/env python3
import time
import threading

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from rosbot_lane_follower_v2.unet_model import UNet
from rosbot_lane_follower_v2.transforms import apply_filters, get_transforms
from rosbot_lane_follower_v2.utils import find_lane_center

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # ─── PARAMETERS ─────────────────────────────────────────────────────
        self.declare_parameter("processing_rate", 10.0)
        self.declare_parameter("roi_y_start",    0.5)
        self.declare_parameter("roi_x_start",    0.0)
        self.declare_parameter("roi_x_end",      1.0)
        self.declare_parameter("debug",          True)
        self.declare_parameter(
            "model_path",
            "/home/wheeltec/ros_ws3/src/rosbot_lane_follower_v2/models/lane_unet.pth"
        )

        prate       = self.get_parameter("processing_rate").value
        self.roi_y  = self.get_parameter("roi_y_start").value
        self.roi_x1 = self.get_parameter("roi_x_start").value
        self.roi_x2 = self.get_parameter("roi_x_end").value
        self.debug  = self.get_parameter("debug").value
        model_path  = self.get_parameter("model_path").value

        # ─── MODEL SETUP ────────────────────────────────────────────────────
        self.bridge   = CvBridge()
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = get_transforms(training=False)

        # ─── ROS I/O ───────────────────────────────────────────────────────
        self.latest_frame = None
        self.frame_lock   = threading.Lock()

        self.create_subscription(
            Image, '/camera/image_raw', self.image_cb, qos_profile=1
        )
        self.lane_pub = self.create_publisher(
            Float32, 'lane_center', qos_profile=1
        )

        # ─── FPS TRACKERS ───────────────────────────────────────────────────
        self.last_cam_time  = time.time()
        self.cam_fps        = 0.0
        self.last_proc_time = time.time()
        self.proc_fps       = 0.0

        if self.debug:
            cv2.namedWindow("LaneFollowerDebug", cv2.WINDOW_NORMAL)
            cv2.startWindowThread()

        # ─── WORKER TIMER ───────────────────────────────────────────────────
        self.create_timer(1.0/prate, self.worker)

    def image_cb(self, msg: Image):
        now = time.time()
        dt  = now - self.last_cam_time
        if dt>0:
            self.cam_fps = 1.0/dt
        self.last_cam_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        with self.frame_lock:
            self.latest_frame = frame

    def worker(self):
        # grab latest frame (drop older ones)
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            self.latest_frame = None

        if frame is None:
            return

        h, w, _ = frame.shape
        # 1) ROI crop
        y1 = int(h * self.roi_y)
        x1 = int(w * self.roi_x1)
        x2 = int(w * self.roi_x2)
        roi = frame[y1:h, x1:x2]

        # 2) Preprocess: Gaussian blur + normalize
        roi_filt = apply_filters(roi)
        inp = self.transform(roi_filt).unsqueeze(0).to(self.device)

        # 3) Inference
        with torch.no_grad():
            out  = self.model(inp)
            pred = torch.sigmoid(out).squeeze().cpu().numpy()

        # 4) Threshold → binary mask
        _, mask = cv2.threshold(pred, 0.5, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        # 5) Compute lane center x
        cx = find_lane_center(mask) + x1

        # 6) Publish
        self.lane_pub.publish(Float32(data=float(cx)))

        # 7) Processing FPS
        now = time.time()
        dt  = now - self.last_proc_time
        if dt>0:
            self.proc_fps = 1.0 / dt
        self.last_proc_time = now

        # 8) Debug overlay
        if self.debug:
            vis = frame.copy()
            cv2.rectangle(vis, (x1, y1), (x2, h), (0,255,0), 2)
            cv2.circle(vis, (int(cx), h-5), 5, (0,0,255), -1)
            cv2.putText(vis, f"CamFPS: {self.cam_fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(vis, f"ProcFPS: {self.proc_fps:.1f}", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("LaneFollowerDebug", vis)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
