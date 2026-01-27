# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:03:37 2025

@author: yzhao
"""

from pathlib import Path

from setuptools import setup


def load_requirements():
    with open(Path(__file__).parent / "requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="pupil-analysis",
    version="0.1.2",
    description="Automated mouse pupil segmentation and diameter analysis using UNet",
    author="Yue Zhao",
    py_modules=["extract_frames", "run_pupil_analysis", "dataset", "unet"],
    install_requires=load_requirements(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "extract-frames=extract_frames:main",
            "run-pupil-analysis=run_pupil_analysis:main",
        ],
    },
)
