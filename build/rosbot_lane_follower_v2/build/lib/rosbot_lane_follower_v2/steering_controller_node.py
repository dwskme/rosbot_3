# steering_controller_node.py - Enhanced with visualization

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import math
import cv2
import numpy as np

from rosbot_lane_follower_v2.utils import PID

class SteeringControllerNode(Node):
    def __init__(self):
        super().__init__('steering_controller_node')

        self.bridge = CvBridge()
        
        # Subscribers
        self.lane_sub = self.create_subscription(
            Float32, 
            '/lane_center', 
            self.lane_callback, 
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10
        )
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.viz_pub = self.create_publisher(Image, '/steering_viz', 10)  # Visualization

        # Control parameters
        self.pid = PID(kp=0.005, ki=0.0, kd=0.002)
        self.last_time = time.time()
        self.image_width = 320  # Visualization width
        self.image_height = 240  # Visualization height
        
        # Safety parameters
        self.max_linear_speed = 0.3
        self.max_angular_speed = 1.0
        self.min_linear_speed = 0.05
        
        # State tracking
        self.last_center_x = None
        self.current_error = 0.0
        self.current_steering = 0.0
        self.current_speed = 0.0
        
        # History for visualization
        self.error_history = []
        self.steering_history = []
        self.max_history = 100
        
        self.get_logger().info("🎮 Steering Controller with Visualization ready!")
        self.get_logger().info(f"📊 View steering visualization at: /steering_viz")

    def create_steering_visualization(self):
        """Create a visualization showing steering status"""
        viz_img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Background
        viz_img.fill(30)  # Dark gray background
        
        # Title
        cv2.putText(viz_img, "STEERING CONTROL", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Lane center visualization
        center_y = 60
        image_center = self.image_width // 2
        
        # Draw road representation
        road_top = center_y - 20
        road_bottom = center_y + 20
        cv2.rectangle(viz_img, (50, road_top), (self.image_width-50, road_bottom), (100, 100, 100), -1)
        
        # Draw center line
        cv2.line(viz_img, (image_center, road_top), (image_center, road_bottom), (255, 255, 0), 2)
        
        # Draw detected lane center
        if self.last_center_x is not None:
            # Scale lane center to visualization width
            scaled_center = int((self.last_center_x / 256.0) * (self.image_width - 100)) + 50
            cv2.circle(viz_img, (scaled_center, center_y), 8, (0, 255, 0), -1)
            
            # Draw error line
            cv2.line(viz_img, (image_center, center_y), (scaled_center, center_y), (255, 0, 0), 3)
        
        # Status text
        y_pos = 100
        cv2.putText(viz_img, f"Lane Center: {self.last_center_x or 'None'}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 20
        cv2.putText(viz_img, f"Error: {self.current_error:.1f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        y_pos += 20
        cv2.putText(viz_img, f"Steering: {self.current_steering:.3f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y_pos += 20
        cv2.putText(viz_img, f"Speed: {self.current_speed:.3f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # PID parameters
        y_pos += 30
        cv2.putText(viz_img, f"PID: kp={self.pid.kp} ki={self.pid.ki} kd={self.pid.kd}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw steering angle indicator
        self.draw_steering_wheel(viz_img, self.image_width - 80, center_y, self.current_steering)
        
        # Draw error history graph
        if len(self.error_history) > 1:
            self.draw_history_graph(viz_img, self.error_history, (255, 0, 0), "Error")
        
        return viz_img

    def draw_steering_wheel(self, img, center_x, center_y, steering_angle):
        """Draw a simple steering wheel indicator"""
        radius = 30
        # Draw wheel circle
        cv2.circle(img, (center_x, center_y), radius, (100, 100, 100), 2)
        
        # Draw steering indicator
        angle_rad = steering_angle * 5  # Scale for visibility
        end_x = int(center_x + radius * 0.8 * np.cos(angle_rad - np.pi/2))
        end_y = int(center_y + radius * 0.8 * np.sin(angle_rad - np.pi/2))
        
        color = (0, 255, 0) if abs(steering_angle) < 0.1 else (0, 255, 255)
        cv2.line(img, (center_x, center_y), (end_x, end_y), color, 3)

    def draw_history_graph(self, img, history, color, label):
        """Draw a simple history graph"""
        if len(history) < 2:
            return
            
        graph_x = 10
        graph_y = 180
        graph_w = 200
        graph_h = 40
        
        # Draw graph background
        cv2.rectangle(img, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), -1)
        
        # Draw zero line
        zero_y = graph_y + graph_h // 2
        cv2.line(img, (graph_x, zero_y), (graph_x + graph_w, zero_y), (100, 100, 100), 1)
        
        # Scale and draw history
        if history:
            max_val = max(abs(min(history)), abs(max(history)), 1.0)
            scale = (graph_h // 2) / max_val
            
            for i in range(1, len(history)):
                x1 = graph_x + int((i-1) * graph_w / len(history))
                x2 = graph_x + int(i * graph_w / len(history))
                y1 = zero_y - int(history[i-1] * scale)
                y2 = zero_y - int(history[i] * scale)
                cv2.line(img, (x1, y1), (x2, y2), color, 1)
        
        # Label
        cv2.putText(img, label, (graph_x, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def lane_callback(self, msg):
        center_x = msg.data
        self.last_center_x = center_x
        
        # Calculate error (negative means lane is to the left, positive means right)
        image_center = 128  # Assuming 256 width / 2
        error = center_x - image_center
        self.current_error = error
        
        # Calculate time difference
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Compute PID control
        steer = self.pid.compute(error, dt)
        self.current_steering = steer
        
        # Create and publish command
        cmd = Twist()
        
        # Adjust speed based on steering angle (slow down for sharp turns)
        abs_steer = abs(steer)
        if abs_steer > 0.5:
            linear_speed = self.min_linear_speed
        else:
            linear_speed = self.max_linear_speed * (1.0 - abs_steer)
            linear_speed = max(linear_speed, self.min_linear_speed)
        
        self.current_speed = linear_speed
        cmd.linear.x = linear_speed
        cmd.angular.z = -steer  # Negative because left turn is +Z in ROS
        
        # Apply limits
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
        
        self.cmd_pub.publish(cmd)
        
        # Update history
        self.error_history.append(error)
        self.steering_history.append(steer)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            self.steering_history.pop(0)
        
        # Create and publish visualization
        viz_img = self.create_steering_visualization()
        viz_msg = self.bridge.cv2_to_imgmsg(viz_img, encoding='bgr8')
        self.viz_pub.publish(viz_msg)
        
        # Enhanced logging with emojis
        direction = "⬅️ LEFT" if steer > 0.1 else "➡️ RIGHT" if steer < -0.1 else "⬆️ STRAIGHT"
        self.get_logger().info(
            f"🎯 Center: {center_x:.1f} | ❌ Error: {error:.1f} | "
            f"🚗 Speed: {cmd.linear.x:.3f} | 🎮 Steer: {cmd.angular.z:.3f} | {direction}"
        )

    def odom_callback(self, msg):
        # Optional: Log current position and velocity
        pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SteeringControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("👋 Steering controller stopped")
    except Exception as e:
        print(f"❌ Steering controller failed: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
