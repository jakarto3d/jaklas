#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import re

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

about = open(join("src", "jaklas", "__about__.py")).read()
version = re.search(r"__version__ ?= ?['\"](.+)['\"]", about).group(1)

setup(
    name="jaklas",
    version=version,
    description="Point cloud writer to las file.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Arnaud Venet",
    author_email="arnaud.venet@jakarto.com",
    url="https://github.com/jakarto3d/jaklas",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=requirements,
    license="Jakarto Licence",
    zip_safe=False,
)
