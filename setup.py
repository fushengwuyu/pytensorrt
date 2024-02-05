# author: sunshine
# datetime:2024/1/8 下午5:43
import setuptools

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
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
