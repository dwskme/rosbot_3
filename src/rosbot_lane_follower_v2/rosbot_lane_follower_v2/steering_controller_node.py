#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from rosbot_lane_follower_v2.utils import PID

class SteeringControllerNode(Node):
    def __init__(self):
        super().__init__('steering_controller_node')

        # ─── PARAMETERS ─────────────────────────────────────────────────────
        self.declare_parameter('kp',              0.5)
        self.declare_parameter('ki',              0.0)
        self.declare_parameter('kd',              0.1)
        self.declare_parameter('max_steer_angle', 0.34)
        self.declare_parameter('camera_offset',   0.0)
        self.declare_parameter('frame_width',   640.0)

        # fetch them
        kp    = self.get_parameter('kp').value
        ki    = self.get_parameter('ki').value
        kd    = self.get_parameter('kd').value
        max_s = self.get_parameter('max_steer_angle').value
        self.camera_offset = self.get_parameter('camera_offset').value
        self.frame_width   = self.get_parameter('frame_width').value

        # ─── PID CONTROLLER ──────────────────────────────────────────────────
        # pass output_limits as a tuple (min, max)
        self.pid = PID(kp, ki, kd, (-max_s, max_s))
        self.current_speed = 0.0

        # ─── ROS SUBSCRIPTIONS & PUBLISHER ──────────────────────────────────
        self.create_subscription(
            Float32, 'lane_center',
            self.lane_cb, qos_profile=1
        )
        self.create_subscription(
            Odometry, 'odom',
            self.odom_cb, qos_profile=1
        )
        self.steer_pub = self.create_publisher(
            AckermannDriveStamped, 'steering_cmd',
            qos_profile=1
        )

        self.get_logger().info(
            f"SteeringControllerNode started (kp={kp}, ki={ki}, kd={kd}, "
            f"limits=[{-max_s}, {max_s}], offset={self.camera_offset})"
        )

    def odom_cb(self, msg: Odometry):
        # update current forward speed
        self.current_speed = msg.twist.twist.linear.x

    def lane_cb(self, msg: Float32):
        # receive lane center x in pixels
        cx = msg.data

        # compute pixel error around image center, then apply camera offset
        error = (cx - (self.frame_width / 2.0)) - self.camera_offset

        # run the PID
        steer = self.pid.update(error)

        # assemble Ackermann cmd
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = steer
        cmd.drive.speed          = self.current_speed

        # publish it
        self.steer_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SteeringControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
