from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'origincar_avoid'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Short-horizon obstacle avoidance and velocity mux for OriginCar.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmd_vel_mux = origincar_avoid.cmd_vel_mux:main',
            'cmd_vel_gate = origincar_avoid.cmd_vel_gate:main',
            'yolo_avoid_controller = origincar_avoid.yolo_avoid_controller:main',
            'yolo_box_viewer = origincar_avoid.yolo_box_viewer:main',
            'teleop_smooth_ros2 = origincar_avoid.teleop_smooth_ros2:main',
        ],
    },
)
