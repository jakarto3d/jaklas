
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()


setup(
    name='las_writer',
    version='0.0.1',
    description="Point cloud writer to las file.",
    long_description=readme,
    author="Arnaud Venet",
    author_email='arnaud.venet@jakarto.com',
    url='https://github.com/jakarto3d/las_writer',
    packages=['las_writer'],
    package_dir={'las_writer': 'src/las_writer'},
    package_data={'las_writer': ['template.las']},
    include_package_data=True,
    install_requires=requirements,
    license="Jakarto Licence",
    zip_safe=False,
)
