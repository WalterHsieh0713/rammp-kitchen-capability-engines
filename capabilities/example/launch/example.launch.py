"""Launches the capability. The image runs this same file, so a native
`ros2 launch` and the container start the same thing.

    ros2 launch example example.launch.py                          # with the orchestrator
    ros2 launch example example.launch.py use_orchestrator:=false  # standalone
    ros2 launch example example.launch.py mock:=true               # no hardware
    ros2 launch example example.launch.py mock:=true mock_urdf:=/robot.urdf  # model for make_sim()
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = ("use_orchestrator", "true"), ("mock", "false")
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default in args]
        + [
            # arm.mock_urdf: the URDF make_sim() loads in mock runs ("" loads none)
            DeclareLaunchArgument("mock_urdf", default_value=""),
            Node(
                package="example",
                executable="example_node",
                output="screen",
                parameters=[
                    {name: ParameterValue(LaunchConfiguration(name), value_type=bool)}
                    for name, _ in args
                ]
                + [
                    {
                        "arm.mock_urdf": ParameterValue(
                            LaunchConfiguration("mock_urdf"), value_type=str
                        )
                    }
                ],
            ),
        ]
    )
