from setuptools import setup, find_packages

setup(
    name="coupled_learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    author="Chris Cai",
    description="Physics-driven coupled learning for adaptive membrane networks",
    python_requires=">=3.8",
)