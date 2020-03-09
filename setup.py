import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="cortado",
    version="1.0-rc3",
    description="High performance ML library with ultra fast XGBoost implementation in pure Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Statfactory/cortado",
    author="Adam Mlocek",
    license="MIT",
    packages=["cortado"],
    include_package_data=False,
    install_requires=["numpy", "numba", "pandas"]
)