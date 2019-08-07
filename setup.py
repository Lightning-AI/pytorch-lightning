#!/usr/bin/env python

# Always prefer setuptools over distutils
from os import path
from setuptools import setup

import pytorch_lightning

PATH_HERE = path.abspath(path.dirname(__file__))

with open(path.join(PATH_HERE, 'requirements.txt'), encoding='utf-8') as fp:
    requirements = [rq.rstrip() for rq in fp.readlines() if not rq.startswith('#')]

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
setup(
    name="pytorch-lightning",
    version=pytorch_lightning.__version__,
    description=pytorch_lightning.__doc__,
    author=pytorch_lightning.__author__,
    author_email=pytorch_lightning.__author_email__,
    url=pytorch_lightning.__homepage__,
    license=pytorch_lightning.__license__,
    packages=['pytorch_lightning'],

    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',

    include_package_data=True,
    zip_safe=False,

    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.6",
    install_requires=requirements,

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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
