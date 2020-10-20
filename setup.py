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
from io import open

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
PATH_ROOT = os.path.dirname(__file__)
builtins.__LIGHTNING_SETUP__ = True

import pytorch_lightning  # noqa: E402


def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def _parse_README_for_badge(text):

    # badge to download
    valid_badge_names = [
        'PyPI - Python Version',
        'PyPI Status',
        'PyPI Status',
        'Conda',
        'DockerHub',
        'codecov',
        'ReadTheDocs',
        'Slack',
        'Discourse status',
        'license',
        'Next Release'
    ]

    for line in text.split('\n'):
        badge_name = re.search(r'^\[!\[(.*?)]', line)

        # check for the badge name
        if badge_name is not None:
            badge_name = badge_name.group(1)

            # check if valid name
            if badge_name in valid_badge_names:

                search_string = fr'\[\!\[{badge_name}]\((.*?)\)]'
                badge_url = re.search(search_string, line)
                # check for badge url
                if badge_url is not None:
                    badge_url = badge_url.group(1)

                # download badge
                saved_badge_name = _download_badges(badge_url, badge_name)

                # replace url with local file path
                replace_string =  f'[![{badge_name}]({saved_badge_name})]'
                text = re.sub(search_string, replace_string, text)

    return text


def _download_badges(url_badge, badge_name):

    base_path = f'docs/source/_images/badges'
    os.makedirs(base_path, exist_ok=True)

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/svg,*/*;q=0.8', }

    # function for saving the badge either in `.png` or `.svg`
    def _save_file(url, save, extension):

        # because there are two badge with name `PyPI Status` the second one is download
        if 'https://pepy.tech/badge/pytorch-lightning' in url_badge:
            save += '_downloads'

        try:
            req = Request(url=url, headers=headers)
            resp = urlopen(req)
        except URLError as err:
            warnings.warn("Error while downloading the badge", UserWarning)
        else:
            save += extension
            with open(save, 'wb') as download_file:
                download_file.write(resp.read())

    save_path = badge_name.replace(' - ', ' ')
    save_path = f"{base_path}/{save_path.replace(' ', '_')}_badge"

    try:
        # always try to download the png versions (some url have an already png version available)
        _save_file(url_badge, save_path, extension='.png')
        return save_path + '.png'
    except HTTPError as err:
        if err.code == 404:
            # save the `.svg`
            url_badge = url_badge.replace('.png', '.svg')
            _save_file(url_badge, save_path, extension='.svg')
            return save_path +'.svg'


def load_long_description():
    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    url = os.path.join(pytorch_lightning.__homepage__, 'raw', pytorch_lightning.__version__, 'docs')
    text = open('README.md', encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace('.svg', '.png')
    # download badge and replace url with local file
    text = _parse_README_for_badge(text)
    return text


# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install pytorch-lightning[dev, docs]`
# From local copy of repo, use like `pip install ".[dev, docs]"`
extras = {
    # 'docs': load_requirements(file_name='docs.txt'),
    'examples': load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='examples.txt'),
    'loggers': load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='loggers.txt'),
    'extra': load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='extra.txt'),
    'test': load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='test.txt')
}
extras['dev'] = extras['extra'] + extras['loggers'] + extras['test']
extras['all'] = extras['dev'] + extras['examples']  # + extras['docs']

# These packages shall be installed only on GPU machines
PACKAGES_GPU_ONLY = (
    'horovod',
)
# create a version for CPU machines
for ex in ('cpu', 'cpu-extra'):
    kw = ex.split('-')[1] if '-' in ex else 'all'
    # filter cpu only packages
    extras[ex] = [pkg for pkg in extras[kw] if not any(pgpu.lower() in pkg.lower() for pgpu in PACKAGES_GPU_ONLY)]

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="pytorch-lightning-nightly",
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
    install_requires=load_requirements(),
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
