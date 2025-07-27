#!/usr/bin/env python
import threading, time
import rospy, cv2, torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from transforms import apply_filters, get_transforms
from unet_model import LaneNet
from postprocess import clean_mask, get_lane_center, hough_lines

class LaneFollowerNode:
    def __init__(self):
        rospy.init_node('lane_follower_node')

        # params
        self.processing_rate = rospy.get_param('~processing_rate', 10)  # Hz
        self.roi_y_start    = rospy.get_param('~roi_y_start', 0.5)
        self.roi_x_start    = rospy.get_param('~roi_x_start', 0.0)
        self.roi_x_end      = rospy.get_param('~roi_x_end',   1.0)
        self.debug          = rospy.get_param('~debug', True)
        self.use_hough      = rospy.get_param('~use_hough', False)
        model_path          = rospy.get_param('~model_path')

        # model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LaneNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = get_transforms(training=False)

        # ROS I/O
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        rospy.Subscriber('/camera/image_raw', Image,
                         self.image_cb, queue_size=1, buff_size=2**24)
        self.lane_pub = rospy.Publisher('lane_center', Float32, queue_size=1)

        # FPS logging
        self.last_cam = time.time()
        self.cam_fps  = 0.0
        self.last_pr  = time.time()
        self.pr_fps   = 0.0

        threading.Thread(target=self._worker, daemon=True).start()
        rospy.spin()

    def image_cb(self, msg):
        now = time.time()
        dt = now - self.last_cam
        if dt>0: self.cam_fps = 1.0/dt
        self.last_cam = now

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.frame_lock:
            self.latest_frame = frame

    def _worker(self):
        rate = rospy.Rate(self.processing_rate)
        while not rospy.is_shutdown():
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                self.latest_frame = None

            if frame is not None:
                h, w, _ = frame.shape
                # 1) ROI crop
                y1 = int(h*self.roi_y_start)
                x1 = int(w*self.roi_x_start)
                x2 = int(w*self.roi_x_end)
                roi = frame[y1:h, x1:x2]

                # 2) Preprocess: Gaussian blur + normalize
                roi_filt = apply_filters(roi)
                inp = self.transform(roi_filt).unsqueeze(0).to(self.device)

                # 3) Inference
                with torch.no_grad():
                    out  = self.model(inp)
                    pred = torch.sigmoid(out).squeeze().cpu().numpy()

                # 4) Postprocess
                mask = clean_mask(pred)
                cx   = get_lane_center(mask) + x1

                # 4b) Optional Hough (not used for center)
                if self.use_hough:
                    lines = hough_lines(mask)
                    # …you could compute an average heading here and publish it…

                # 5) Publish center
                self.lane_pub.publish(Float32(data=cx))

                # 6) Proc FPS
                now = time.time()
                dtp = now - self.last_pr
                if dtp>0: self.pr_fps = 1.0/dtp
                self.last_pr = now

                # 7) Debug viz
                if self.debug:
                    vis = frame.copy()
                    cv2.rectangle(vis, (x1, y1), (x2, h), (0,255,0), 2)
                    cv2.circle(vis, (int(cx), h-5), 5, (0,0,255), -1)
                    cv2.putText(vis, f'CamFPS: {self.cam_fps:.1f}', (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(vis, f'ProcFPS: {self.pr_fps:.1f}', (10,70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.imshow('LaneFollowerDebug', vis)
                    cv2.waitKey(1)

            rate.sleep()

if __name__=='__main__':
    try:
        LaneFollowerNode()
    except rospy.ROSInterruptException:
        pass
