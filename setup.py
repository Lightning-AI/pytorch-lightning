#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pytorch-lightning',
      version='0.0.2',
      description='Rapid research framework',
      author='',
      author_email='',
      url='https://github.com/williamFalcon/pytorch-lightning',
      install_requires=['test-tube', 'torch', 'tqdm'],
      packages=find_packages()
      )
