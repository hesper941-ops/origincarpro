from glob import glob
from setuptools import find_packages, setup


package_name = "roadblock_localization"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OriginCar Team",
    maintainer_email="maintainer@example.com",
    description="YOLO roadblock bbox to metric vehicle-ground localization",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "roadblock_ground_localizer = roadblock_localization.roadblock_ground_localizer:main",
        ],
    },
)
