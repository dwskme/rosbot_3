#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from utils import PID

class SteeringControllerNode:
    def __init__(self):
        rospy.init_node('steering_controller_node')
        kp = rospy.get_param('~kp', 0.5)
        ki = rospy.get_param('~ki', 0.0)
        kd = rospy.get_param('~kd', 0.1)
        max_steer = rospy.get_param('~max_steer_angle', 0.34)
        self.pid = PID(kp, ki, kd, output_limits=(-max_steer, max_steer))

        self.camera_offset = rospy.get_param('~camera_offset', 0.0)
        self.frame_width   = rospy.get_param('~frame_width', 640)
        self.current_speed = 0.0

        rospy.Subscriber('lane_center', Float32, self.lane_cb, queue_size=1)
        rospy.Subscriber('odom',       Odometry,   self.odom_cb)
        self.steer_pub = rospy.Publisher('steering_cmd',
                                         AckermannDriveStamped,
                                         queue_size=1)
        rospy.spin()

    def odom_cb(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def lane_cb(self, msg):
        cx = msg.data
        error = (cx - (self.frame_width/2.0)) - self.camera_offset
        steer = self.pid.update(error)

        cmd = AckermannDriveStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.drive.steering_angle = steer
        cmd.drive.speed = self.current_speed
        self.steer_pub.publish(cmd)

if __name__=='__main__':
    try:
        SteeringControllerNode()
    except rospy.ROSInterruptException:
        pass
