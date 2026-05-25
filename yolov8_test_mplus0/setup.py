from setuptools import setup
from glob import glob
import os

package_name = 'yolov8_test_mplus0'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
            ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='YOLOv8 test package for intelligent car camera and detection visualization.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_box_viewer = yolov8_test_mplus0.yolo_box_viewer:main',
        ],
    },
)
