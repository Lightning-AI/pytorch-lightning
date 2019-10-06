#!/usr/bin/env python

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# https://packaging.python.org/guides/single-sourcing-package-version/

# http://blog.ionelmc.ro/2014/05/25/python-packaging/

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name='pytorch-lightning',
    version='0.5.1.3',
    description='The Keras for ML researchers using PyTorch',
    author='William Falcon',
    author_email='waf2107@columbia.edu',
    url='https://github.com/williamFalcon/pytorch-lightning',
    download_url='https://github.com/williamFalcon/pytorch-lightning',
    license='Apache-2',
    packages=find_packages(),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.2.0',
        'tqdm>=4.35.0',
        'test-tube>=0.6.9',
        'pandas>=0.20.3',
    ],
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
