from setuptools import setup
import os
from glob import glob

package_name = 'rosbot_lane_follower_v2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [
            'resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'opencv-python',
        'albumentations',
        'numpy'
    ],
    zip_safe=True,
    maintainer='sarvesh',
    maintainer_email='sarvesh@example.com',
    description='ROS 2 package for lane detection and autonomous control using deep learning (U-Net)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follower_node = rosbot_lane_follower_v2.lane_follower_node:main',
            'steering_controller_node = rosbot_lane_follower_v2.steering_controller_node:main',
        ],
    },
)

