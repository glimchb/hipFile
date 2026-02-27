#!/usr/bin/env python3
"""Setup script for hipfile Python bindings."""

from setuptools import setup, find_packages

setup(
    name="hipfile",
    version="0.1.0",
    description="Python bindings for AMD hipFile – GPU-direct storage on ROCm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AMD ROCm Community",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "rocm": ["amdsmi"],
        "torch": ["torch"],
        "dev": ["pytest", "pytest-cov"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Hardware",
    ],
    keywords=["amd", "rocm", "hip", "gpu", "storage", "io", "gpudirect"],
)
