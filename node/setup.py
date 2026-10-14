from setuptools import setup
from glob import glob

package_name = "mpcc_online"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="agpoks",
    maintainer_email="agnes.poks@gmail.com",
    description="Online-tuned MPCC deploy node for the RC car (reuses mpcc_tuning + acados).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mpcc_online = mpcc_online.ros2_node:main",
        ],
    },
)
