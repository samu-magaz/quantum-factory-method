# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='qfm',
    version='0.1.0',
    description='Examples for paper "Quantum Factory Method: A Design for Implementing Different Quantum Libraries"',
    long_description=readme,
    author='Samuel Magaz-Romero',
    author_email='s.magazr@udc.es',
    url='https://github.com/samu-magaz/quantum-factory-method',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

