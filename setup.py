#!/usr/bin/env python

import os
from io import open
# Always prefer setuptools over distutils
from setuptools import setup, find_packages

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/

builtins.__LIGHTNING_SETUP__ = True

import pytorch_lightning  # noqa: E402


def load_long_description():
    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    url = os.path.join(pytorch_lightning.__homepage__, 'raw', pytorch_lightning.__version__, 'docs')
    text = open('README.md', encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace('.svg', '.png')
    return text


extras = {
    'base': [
        'numpy>=1.16.4',
        'torch>=1.3',
        'tensorboard>=1.14',
        'future>=0.17.1',
        'PyYAML>=5.1',
        'tqdm>=4.41.0'
    ],
    'docs': [
        'sphinx>=2.0, <3.0',
        'recommonmark',
        'm2r',
        'nbsphinx',
        'pandoc',
        'docutils',
        'sphinxcontrib-fulltoc',
        'sphinxcontrib-mockautodoc',

        # Splitting line here to adhere to style standards
        'pt_lightning_sphinx_theme @ '
        'git+git://github.com/PytorchLightning/lightning_sphinx_theme.git'
        '@b635a2a21883f830d4e063ede6df64f0b5773683#egg=pt-lightning-sphinx-theme'

        'sphinx-autodoc-typehints',
        'sphinx-paramlinks<0.4.0'
    ],
    'examples': [
        'torchvision>=0.4.0, <0.7',
        'gym>=0.17.0'
    ],
    'extra': [
        'neptune-client>=0.4.109',
        'comet-ml>=1.0.56',
        'mlflow>=1.0.0',
        'test_tube>=0.7.5',
        'wandb>=0.8.21',
        'matplotlib>=3.1.1',
        'horovod>=0.19.2',
        'omegaconf>=2.0.0',
        'scikit-learn>=0.20.0',
        'torchtext>=0.3.1, <0.7',
        'onnx>=1.7.0',
        'onnxruntime>=1.3.0'
    ],
    'test': [
        'coverage',
        'codecov>=2.1',
        'pytest>=3.0.5',
        'pytest-cov',
        'pytest-flake8',
        'flake8',
        'flake8-black',
        'check-manifest',
        'twine==1.13.0',
        'scikit-image',
        'black==19.10b0',
        'pre-commit>=1.0',
        'cloudpickle>=1.2',
        'nltk>=3.3'
    ]
}
extras['dev'] = extras['base'] + extras['extra'] + extras['test']
extras['all'] = extras['dev'] + extras['examples'] + extras['docs']
base_requirements = extras.pop('base')

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name='pytorch-lightning',
    version=pytorch_lightning.__version__,
    description=pytorch_lightning.__docs__,
    author=pytorch_lightning.__author__,
    author_email=pytorch_lightning.__author_email__,
    url=pytorch_lightning.__homepage__,
    download_url='https://github.com/PyTorchLightning/pytorch-lightning',
    license=pytorch_lightning.__license__,
    packages=find_packages(exclude=['tests', 'tests/*', 'benchmarks']),

    long_description=load_long_description(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,

    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=base_requirements,
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
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
