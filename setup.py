#!/usr/bin/env python3
# coding=utf-8

"""
setup for sunglint correction module
"""

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="sungc",
    description=(
        "Sunglint correction algorithms for Sentinel-2a/b, "
        "Worldview-2, Landsat-8 OLI, Landsat-7 ETM+ & Landsat-5 TM"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Rodrigo A. Garcia",
    author_email="earth.observation@ga.gov.au",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={"": ["*.json", "*.yaml"]},
    license="Apache Software License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/GeoscienceAustralia/sun-glint-correction",
    install_requires=[
        "scipy",
        "fiona",
        "numpy",
        "pillow",
        "shapely",
        "rasterio",
        "matplotlib",
    ],
)
