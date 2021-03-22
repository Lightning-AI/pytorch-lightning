#!/usr/bin/env python
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# Always prefer setuptools over distutils
import sys

from setuptools import find_packages, setup

try:
    from pytorch_lightning import info, setup_tools
except ImportError:
    # alternative https://stackoverflow.com/a/67692/4521646
    sys.path.append("pytorch_lightning")
    import info
    import setup_tools

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, 'requirements')

# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install pytorch-lightning[dev, docs]`
# From local copy of repo, use like `pip install ".[dev, docs]"`
extras = {
    # 'docs': load_requirements(file_name='docs.txt'),
    'examples': setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name='examples.txt'),
    'loggers': setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name='loggers.txt'),
    'extra': setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name='extra.txt'),
    'test': setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name='test.txt')
}
extras['dev'] = extras['extra'] + extras['loggers'] + extras['test']
extras['all'] = extras['dev'] + extras['examples']  # + extras['docs']

# These packages shall be installed only on GPU machines
PACKAGES_GPU_ONLY = ['horovod']
# create a version for CPU machines
for ex in ('cpu', 'cpu-extra'):
    kw = ex.split('-')[1] if '-' in ex else 'all'
    # filter cpu only packages
    extras[ex] = [pkg for pkg in extras[kw] if not any(pgpu.lower() in pkg.lower() for pgpu in PACKAGES_GPU_ONLY)]

long_description = setup_tools._load_readme_description(
    _PATH_ROOT,
    homepage=info.__homepage__,
    version=info.__version__,
)

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="pytorch-lightning",
    version=info.__version__,
    description=info.__docs__,
    author=info.__author__,
    author_email=info.__author_email__,
    url=info.__homepage__,
    download_url='https://github.com/PyTorchLightning/pytorch-lightning',
    license=info.__license__,
    packages=find_packages(exclude=['tests', 'tests/*', 'benchmarks', 'legacy', 'legacy/*']),
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=setup_tools._load_requirements(_PATH_ROOT),
    extras_require=extras,
    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/pytorch-lightning/issues",
        "Documentation": "https://pytorch-lightning.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PyTorchLightning/pytorch-lightning",
    },
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
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
