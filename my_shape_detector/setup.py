from setuptools import setup

package_name = 'my_shape_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy', 'cv_bridge', 'sensor_msgs', 'image_transport'],
    zip_safe=True,
    author='nam27',
    author_email='nam27@example.com',
    description='Shape detection node for ROS 2',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'shape_detect = my_shape_detector.shape_detect_node:main',  # Ensure this points to the main function of your node
        ],
    },
)
