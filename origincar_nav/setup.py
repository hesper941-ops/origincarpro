from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'origincar_nav'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Competition navigation state machine and odometry target tracker.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_tracker = origincar_nav.target_tracker:main',
            'task_manager = origincar_nav.task_manager:main',
            'semantic_map_visualizer = origincar_nav.semantic_map_visualizer:main',
        ],
    },
)
