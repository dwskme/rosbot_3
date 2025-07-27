# lane_follower_node.py - Shows ONLY lanes, blacks out everything else

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import os
from rosbot_lane_follower_v2.unet_model import UNet
from rosbot_lane_follower_v2.transforms import apply_filters, get_transforms
from rosbot_lane_follower_v2.utils import find_lane_center

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model path - correct path for wheeltec user
        model_path = self.declare_parameter(
            "model_path", 
            "/home/wheeltec/ros_ws3/src/rosbot_lane_follower_v2/models/lane_unet.pth"  
        ).get_parameter_value().string_value
        
        self.get_logger().info(f"Loading model from: {model_path}")

        # Load model
        try:
            self.model = UNet().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info("✅ Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {str(e)}")
            return

        self.transform = get_transforms()

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10)

        # Publishers
        self.lane_only_pub = self.create_publisher(Image, '/lane_debug', 10)  # Shows ONLY lanes
        self.center_pub = self.create_publisher(Float32, '/lane_center', 10)
        
        self.get_logger().info("🚗 Lane Follower Node ready - will show ONLY lanes!")

    def create_lane_only_image(self, original_image, mask):
        """Create image showing ONLY lanes (everything else black)"""
        # Create black background
        lane_only = np.zeros_like(original_image)
        
        # Apply mask to show only lane areas
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
        
        # Where mask is white (lanes detected), show original image
        lane_mask = mask > 0
        lane_only[lane_mask] = original_image[lane_mask]
        
        return lane_only

    def apply_hough_lines(self, mask):
        """Apply Hough transform and return lines"""
        # Focus on lower half for better lane detection
        h, w = mask.shape
        roi = mask[h//2:, :]
        
        # Edge detection
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=20,
            minLineLength=30,
            maxLineGap=10
        )
        
        # Create line image
        line_img = np.zeros_like(mask)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Adjust coordinates back to full image
                cv2.line(line_img, (x1, y1 + h//2), (x2, y2 + h//2), 255, 3)
        
        return line_img

    def listener_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            original_height, original_width = cv_image.shape[:2]
            
        except Exception as e:
            self.get_logger().error(f"❌ CvBridge error: {str(e)}")
            return

        try:
            # Preprocess image
            filtered = apply_filters(cv_image)
            
            # Transform for model
            augmented = self.transform(image=filtered)
            input_tensor = augmented['image'].unsqueeze(0).to(self.device)

            # Model prediction
            with torch.no_grad():
                pred = self.model(input_tensor)

            # Convert prediction to binary mask
            mask = pred.squeeze().cpu().numpy()
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Resize mask to original image size
            binary_mask_resized = cv2.resize(binary_mask, (original_width, original_height))
            
            # Apply Hough lines to enhance lane detection
            hough_lines = self.apply_hough_lines(binary_mask_resized)
            
            # Combine original mask with Hough lines
            combined_mask = cv2.bitwise_or(binary_mask_resized, hough_lines)
            
            # Create lane-only image (black background, only lanes visible)
            lane_only_image = self.create_lane_only_image(cv_image, combined_mask)
            
            # Find lane center from the combined mask
            center_x = find_lane_center(combined_mask)
            
            if center_x is not None:
                # Draw lane center on lane-only image
                cv2.circle(lane_only_image, (center_x, original_height//2), 8, (0, 255, 255), -1)  # Yellow center
                cv2.line(lane_only_image, (center_x, 0), (center_x, original_height), (0, 255, 255), 2)
                
                # Add text showing center position
                cv2.putText(lane_only_image, f"Center: {center_x}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Publish center position
                center_msg = Float32()
                center_msg.data = float(center_x)
                self.center_pub.publish(center_msg)
                
                self.get_logger().info(f"🎯 Lane center detected at: {center_x}")
            else:
                # Add "NO LANES" text
                cv2.putText(lane_only_image, "NO LANES DETECTED", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.get_logger().warn("⚠️  No lanes detected")

            # Add image dimensions for reference
            cv2.putText(lane_only_image, f"Size: {original_width}x{original_height}", 
                       (10, original_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Publish lane-only image
            lane_msg = self.bridge.cv2_to_imgmsg(lane_only_image, encoding='bgr8')
            lane_msg.header = msg.header
            self.lane_only_pub.publish(lane_msg)

        except Exception as e:
            self.get_logger().error(f"❌ Processing error: {str(e)}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LaneFollowerNode()
        node.get_logger().info("🚀 Starting lane-only detection...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("👋 Shutting down...")
    except Exception as e:
        print(f"❌ Node failed: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
