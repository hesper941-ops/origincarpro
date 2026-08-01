from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'person_board_detection'


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'prompts'),
            glob('prompts/*.txt'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gxb',
    maintainer_email='root@localhost',
    description='RDK X5 person board realtime detector',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            (
                'person_board_detector = '
                'person_board_detection.person_board_detector:main'
            ),
            (
                'person_board_mock_llm_worker = '
                'person_board_detection.mock_llm_worker:main'
            ),
            (
                'person_board_display = '
                'person_board_detection.person_board_display:main'
            ),
            (
                'person_board_qwen_vl_worker = '
                'person_board_detection.qwen_vl_worker:main'
            ),
        ],
    },
)
