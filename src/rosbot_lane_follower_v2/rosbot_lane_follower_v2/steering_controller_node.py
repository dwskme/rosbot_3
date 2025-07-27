#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from utils import PID

class SteeringControllerNode(Node):
    def __init__(self):
        super().__init__('steering_controller_node')

        # parameters
        self.declare_parameter('kp', 0.5)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.1)
        self.declare_parameter('max_steer_angle', 0.34)
        self.declare_parameter('camera_offset', 0.0)
        self.declare_parameter('frame_width', 640.0)

        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        max_s = self.get_parameter('max_steer_angle').value
        self.camera_offset = self.get_parameter('camera_offset').value
        self.frame_width   = self.get_parameter('frame_width').value

        self.pid = PID(kp, ki, kd, output_limits=(-max_s, max_s))
        self.current_speed = 0.0

        # subs & pub
        self.create_subscription(
            Float32, 'lane_center',
            self.lane_cb, qos_profile=1
        )
        self.create_subscription(
            Odometry, 'odom',
            self.odom_cb, qos_profile=1
        )
        self.steer_pub = self.create_publisher(
            AckermannDriveStamped, 'steering_cmd', qos_profile=1
        )

    def odom_cb(self, msg: Odometry):
        self.current_speed = msg.twist.twist.linear.x

    def lane_cb(self, msg: Float32):
        cx = msg.data
        error = (cx - (self.frame_width/2.0)) - self.camera_offset
        steer = self.pid.update(error)

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = steer
        cmd.drive.speed = self.current_speed
        self.steer_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SteeringControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
