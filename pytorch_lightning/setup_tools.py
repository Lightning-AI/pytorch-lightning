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
import re
import warnings
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pytorch_lightning import __homepage__, __version__, PROJECT_ROOT

_PATH_BADGES = os.path.join('.', 'docs', 'source', '_images', 'badges')
# badge to download
_DEFAULT_BADGES = [
    'Conda',
    'DockerHub',
    'codecov',
    'ReadTheDocs',
    'Slack',
    'Discourse status',
    'license',
]


def _load_requirements(path_dir: str , file_name: str = 'requirements.txt', comment_char: str = '#') -> List[str]:
    """Load requirements from a file

    >>> _load_requirements(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
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


def _parse_for_badge(
        text: str,
        release_url: str = None,
        path_badges: str = _PATH_BADGES,
        badge_names: Iterable = _DEFAULT_BADGES,
) -> str:
    """ Returns the new parsed text with url change with local downloaded files

    >>> _parse_for_badge('Some text here... '  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ...     '[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda)]'
    ...     '(https://anaconda.org/conda-forge/pytorch-lightning) and another text later')
    'Some text here...
     [![Conda](...docs...source..._images...badges...Conda_badge.svg)](https://anaconda.org/conda-forge/pytorch-lightning)
     and another text later'
    >>> import shutil
    >>> shutil.rmtree(_PATH_BADGES)
    """
    for line in text.split(os.linesep):
        search_string = r'\[\!\[(.*)]\((.*)\)]'
        match = re.search(search_string, line)
        if match is None:
            continue

        badge_name, badge_url = match.groups()
        # check if valid name
        if badge_name not in badge_names:
            continue

        # download badge
        badge_path = _download_badge(badge_url, badge_name, path_badges)
        if release_url:
            # https://github.com/Borda/pytorch-lightning/releases/download/1.1.0a6/codecov_badge.png
            badge_fname = os.path.basename(badge_path)
            badge_path = os.path.join(release_url, badge_fname)

        # replace url with local file path
        text = text.replace(f'[![{badge_name}]({badge_url})]', f'[![{badge_name}]({badge_path})]')

    return text


def _save_file(url_badge: str, save_path: str, extension: str, headers: dict) -> None:
    """function for saving the badge either in `.png` or `.svg`"""

    # because there are two badge with name `PyPI Status` the second one is download
    if 'https://pepy.tech/badge/pytorch-lightning' in url_badge:
        save_path += '_downloads'

    try:
        req = Request(url=url_badge, headers=headers)
        resp = urlopen(req)
    except URLError:
        warnings.warn("Error while downloading the badge", UserWarning)
    else:
        save_path += extension
        with open(save_path, 'wb') as download_file:
            download_file.write(resp.read())


def _download_badge(url_badge: str, badge_name: str, target_dir: str) -> str:
    """Download badge from url

    >>> path_img = _download_badge('https://img.shields.io/pypi/pyversions/pytorch-lightning',
    ...                            'PyPI - Python Version', '.')
    >>> os.path.isfile(path_img)
    True
    >>> path_img  # doctest: +ELLIPSIS
    '...PyPI_Python_Version_badge.png'
    >>> os.remove(path_img)
    """
    os.makedirs(target_dir, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/svg,*/*;q=0.8',
    }

    save_path = badge_name.replace(' - ', ' ')
    save_path = os.path.join(target_dir, f"{save_path.replace(' ', '_')}_badge")

    if "?" in url_badge and ".png" not in url_badge:
        _save_file(url_badge, save_path, extension='.svg', headers=headers)
        return save_path + '.svg'
    else:
        try:
            # always try to download the png versions (some url have an already png version available)
            _save_file(url_badge, save_path, extension='.png', headers=headers)
            return save_path + '.png'
        except HTTPError as err:
            if err.code == 404:
                # save the `.svg`
                url_badge = url_badge.replace('.png', '.svg')
                _save_file(url_badge, save_path, extension='.svg', headers=headers)
                return save_path + '.svg'
            return ''


def _load_long_description(path_dir: str) -> str:
    """Load readme as decribtion

    >>> _load_long_description(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<div align="center">...'
    """
    path_readme = os.path.join(path_dir, "README.md")
    text = open(path_readme, encoding="utf-8").read()

    # drop images from readme
    text = text.replace('![PT to PL](docs/source/_images/general/pl_quick_start_full_compressed.gif)', '')

    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(__homepage__, "raw", __version__)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we reger some other sources with particular path to docs
    text = text.replace("docs/source/_images/", f"{os.path.join(github_source_url, 'docs/source/_images/')}")

    # readthedocs badge
    text = text.replace('badge/?version=stable', f'badge/?version={__version__}')
    text = text.replace('pytorch-lightning.readthedocs.io/en/stable/', f'pytorch-lightning.readthedocs.io/en/{__version__}')
    # codecov badge
    text = text.replace('/branch/master/graph/badge.svg', f'/release/{__version__}/graph/badge.svg')
    # replace github badges for release ones
    text = text.replace('badge.svg?branch=master&event=push', f'badge.svg?tag={__version__}')

    skip_begin = r'<!-- following section will be skipped from PyPI description -->'
    skip_end = r'<!-- end skipping PyPI description -->'
    # todo: wrap content as commented description
    text = re.sub(rf"{skip_begin}.+?{skip_end}", '<!--  -->', text, flags=re.IGNORECASE + re.DOTALL)

    # # https://github.com/Borda/pytorch-lightning/releases/download/1.1.0a6/codecov_badge.png
    # github_release_url = os.path.join(__homepage__, "releases", "download", __version__)
    # # download badge and replace url with local file
    # text = _parse_for_badge(text, github_release_url)
    return text
