#!/usr/bin/env python3
import time
import threading

import cv2
import torch
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from transforms import apply_filters, get_transforms
from unet_model import LaneNet
from postprocess import clean_mask, get_lane_center, hough_lines

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # parameters
        self.declare_parameter('processing_rate', 10.0)
        self.declare_parameter('roi_y_start',    0.5)
        self.declare_parameter('roi_x_start',    0.0)
        self.declare_parameter('roi_x_end',      1.0)
        self.declare_parameter('debug',          True)
        self.declare_parameter('use_hough',      False)
        self.declare_parameter('model_path',     '')

        prate       = self.get_parameter('processing_rate').value
        self.roi_y  = self.get_parameter('roi_y_start').value
        self.roi_x1 = self.get_parameter('roi_x_start').value
        self.roi_x2 = self.get_parameter('roi_x_end').value
        self.debug  = self.get_parameter('debug').value
        self.use_hough = self.get_parameter('use_hough').value
        model_path  = self.get_parameter('model_path').value

        # model init
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LaneNet().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        self.transform = get_transforms(training=False)

        # ROS interfaces
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # subscriptions & pubs
        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_cb, qos_profile=1
        )
        self.lane_pub = self.create_publisher(
            Float32, 'lane_center', qos_profile=1
        )

        # FPS tracking
        self.last_cam = time.time()
        self.cam_fps  = 0.0
        self.last_pr  = time.time()
        self.pr_fps   = 0.0

        # worker timer at fixed rate
        self.create_timer(1.0/prate, self.worker)

    def image_cb(self, msg: Image):
        now = time.time()
        dt = now - self.last_cam
        if dt>0:
            self.cam_fps = 1.0/dt
        self.last_cam = now

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.frame_lock:
            self.latest_frame = frame

    def worker(self):
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            self.latest_frame = None

        if frame is None:
            return

        h, w, _ = frame.shape
        # ROI crop
        y1 = int(h * self.roi_y)
        x1 = int(w * self.roi_x1)
        x2 = int(w * self.roi_x2)
        roi = frame[y1:h, x1:x2]

        # preprocess + infer
        roi = apply_filters(roi)
        inp = self.transform(roi).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out  = self.model(inp)
            pred = torch.sigmoid(out).squeeze().cpu().numpy()

        # postprocess
        mask = clean_mask(pred)
        cx   = get_lane_center(mask) + x1

        # optional hough (not used for center)
        if self.use_hough:
            lines = hough_lines(mask)
            # …compute heading if desired…

        # publish
        self.lane_pub.publish(Float32(data=cx))

        # processing FPS
        now = time.time()
        dtp = now - self.last_pr
        if dtp>0:
            self.pr_fps = 1.0/dtp
        self.last_pr = now

        # debug overlay
        if self.debug:
            vis = frame.copy()
            cv2.rectangle(vis, (x1, y1), (x2, h), (0,255,0), 2)
            cv2.circle(vis,(int(cx), h-5),5,(0,0,255),-1)
            cv2.putText(vis,f'CamFPS:{self.cam_fps:.1f}',(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(vis,f'ProcFPS:{self.pr_fps:.1f}',(10,70),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.imshow('LaneFollowerDebug',vis)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
