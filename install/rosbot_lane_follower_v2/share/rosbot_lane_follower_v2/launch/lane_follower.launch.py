# launch/lane_follower_launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
import os

def generate_launch_description():
    return LaunchDescription([
        # Lane detection node
        Node(
            package='rosbot_lane_follower_v2',
            executable='lane_follower_node',
            name='lane_follower_node',
            output='screen',
            parameters=[{
                'model_path': '/home/wheeltec/ros_ws3/src/rosbot_lane_follower_v2/models/lane_unet.pth'
            }]
        ),
        
        # Steering controller node  
        Node(
            package='rosbot_lane_follower_v2',
            executable='steering_controller_node',
            name='steering_controller_node',
            output='screen'
        ),
        
        # Auto-launch image viewer for lane detection after 3 seconds
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view', '/lane_debug'],
                    output='screen'
                )
            ]
        )
    ])
