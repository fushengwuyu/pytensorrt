# author: sunshine
# datetime:2024/1/8 下午5:43
import setuptools
from setuptools import find_namespace_packages, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytrt",
    version="0.1",
    author="sunshine",
    author_email="",
    description="tensorrt infer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fushengwuyu/pyencryption",
    packages=find_namespace_packages(),
    package_data={
        'pytrt': ['./_lib/*.so'],  # 使用相对于基本目录的路径
    },
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
