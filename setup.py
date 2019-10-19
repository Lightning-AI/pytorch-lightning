#!/usr/bin/env python

import os
# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/

PATH_ROOT = os.path.dirname(__file__)

def get_package_info(name):
    import pytorch_lightning
    info = vars(pytorch_lightning)
    info = {k: info[k] for k in info if k.startswith('__')}
    return info[name]


def load_requirements(path_dir=PATH_ROOT):
    with open(os.path.join(path_dir, 'requirements.txt'), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    # TODO: you may filer also comments
    return lines

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name='pytorch-lightning',
    version=get_package_info('__version__'),
    description=get_package_info('__docs__'),
    author=get_package_info('__author__'),
    author_email=get_package_info('__author_email__'),
    url=get_package_info('__homepage__'),
    download_url='https://github.com/williamFalcon/pytorch-lightning',
    license=get_package_info('__license__'),
    packages=find_packages(exclude=['examples']),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[
        'numpy',
        'torch',
        'tqdm',  # used in trainer
        'pandas',
    ],
    install_requires=load_requirements(PATH_ROOT),
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
