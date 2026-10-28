from setuptools import setup

package_name = "capability_engine"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    py_modules=["rammp_arm"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)
