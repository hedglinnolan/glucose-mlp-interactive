"""
Setup script for the interactive regression model trainer.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="regression-model-trainer",
    version="1.0.0",
    author="Nolan Hedglin",
    description="Interactive web application for training and comparing regression models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hedglinnolan/glucose-mlp-interactive",
    py_modules=["app", "data_processor", "models", "visualizations"],
    install_requires=requirements,
    python_requires=">=3.8,<3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
