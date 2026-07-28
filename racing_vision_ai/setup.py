from setuptools import setup

package_name = 'racing_vision_ai'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/vision_ai_config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Machao615',
    maintainer_email='machao615@foxmail.com',
    description='Racing car vision AI package for image analysis using Volcengine AI model',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_ai_node = racing_vision_ai.vision_ai_node:main',
        ],
    },
)
