#!/usr/bin/env python
"""Setup script for PMF package — flat structure with package_dir."""

from setuptools import setup

setup(
    name="pmf",
    version="0.1.0",
    author="vLAR Group",
    description="Power-spectrum Metric for Frequency-domain video similarity",
    url="https://github.com/vLAR-group/PhysInOne",
    
    # Critical: declare the package and map it to current directory
    packages=["pmf"],
    package_dir={"pmf": "."},
    
    python_requires=">=3.11",
    install_requires=["torch>=1.10.0"],
    
    license="MIT",
    keywords=["video-similarity", "fft", "pytorch", "physics-aware"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)