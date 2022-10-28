#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements_dev.txt") as fp:
    requirements = fp.read()

test_requirements = ["pytest>=3"]

setup(
    author="Vivek Khimani",
    author_email="vivekkhimani07@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    description="A Python library for rapid prototyping, experimenting, and logging of federated learning using state-of-the-art models and datasets. Built using PyTorch and PyTorch Lightning.",
    entry_points={"console_scripts": ["torchfl=torchfl.cli:main"]},
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="torchfl",
    name="torchfl",
    packages=find_packages(include=["torchfl", "torchfl.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/vivekkhimani/torchfl",
    version="0.1.6",
    zip_safe=False,
)
