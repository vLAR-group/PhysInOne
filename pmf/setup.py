#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import re

# --- Metadata Helpers ---
def get_version():
    # Optional: read version from a file or hardcode it
    return "0.1.0"

def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Power Spectrum-based Metric for Video Similarity (PMF)"

setup(
    name="pmf",  # Changed to avoid namespace conflicts with existing 'pmf' packages
    version=get_version(),
    author="vLAR Group",
    long_description=read_long_description(),
    url="https://github.com/vLAR-group/PhysInOne.git",
    packages=["pmf"],
    package_dir={"pmf": "."},  # Tell setuptools: the 'pmf' package is in current dir,
    python_requires=">=3.11",
    install_requires=[
        "torch>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "ruff",
            "mypy",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    license="MIT",
    keywords="video-similarity, fft, pmf, pytorch, metric",
)