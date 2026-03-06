from setuptools import setup, find_packages

setup(
    name="complexity-i64",
    version="0.1.0",
    description="Integer-native Complexity architecture — INT8 by design",
    author="INL",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
    ],
)
