from setuptools import setup
from glob import glob
import os

package_name = 'origincar_nav'

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
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'),
            glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Navigation bringup package for origincar',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # EXECUTABLE_NAME = MODULE_NAME.FILE_NAME:MAIN_FUNCTION
            'map_odom_tf_publisher = origincar_nav.map_odom_tf_publisher:main',
            'waypoint_manager = origincar_nav.waypoint_manager:main',
        ],
    },
)