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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pytorch_lightning import PROJECT_ROOT, __homepage__, __version__

_PATH_BADGES = os.path.join('.', 'docs', 'source', '_images', 'badges')
# badge to download
_DEFAULT_BADGES = [
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


def _load_requirements(path_dir, file_name='requirements.txt', comment_char='#'):
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


def _parse_for_badge(text: str, path_badges: str = _PATH_BADGES, badge_names: list = _DEFAULT_BADGES):
    """ Returns the new parsed text with url change with local downloaded files

    >>> _parse_for_badge('Some text here... '  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ...     '[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)]'
    ...     '(https://pypi.org/project/pytorch-lightning/) and another text later')
    'Some text here...
     [![PyPI - Python Version](...docs...source..._images...badges...PyPI_Python_Version_badge.png)](https://pypi.org/project/pytorch-lightning/)
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
        saved_badge_name = _download_badge(badge_url, badge_name, path_badges)

        # replace url with local file path
        text = text.replace(f'[![{badge_name}]({badge_url})]', f'[![{badge_name}]({saved_badge_name})]')

    return text


def _save_file(url_badge, save, extension, headers):
    """function for saving the badge either in `.png` or `.svg`"""

    # because there are two badge with name `PyPI Status` the second one is download
    if 'https://pepy.tech/badge/pytorch-lightning' in url_badge:
        save += '_downloads'

    try:
        req = Request(url=url_badge, headers=headers)
        resp = urlopen(req)
    except URLError:
        warnings.warn("Error while downloading the badge", UserWarning)
    else:
        save += extension
        with open(save, 'wb') as download_file:
            download_file.write(resp.read())


def _download_badge(url_badge, badge_name, target_dir):
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


def _load_long_description(path_dir):
    """Load readme as decribtion

    >>> _load_long_description(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<div align="center">...'
    >>> import shutil
    >>> shutil.rmtree(_PATH_BADGES)
    """
    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    url = os.path.join(__homepage__, 'raw', __version__, 'docs')
    path_readme = os.path.join(path_dir, 'README.md')
    text = open(path_readme, encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them with PNG
    text = text.replace('.svg', '.png')
    # download badge and replace url with local file
    text = _parse_for_badge(text)
    return text
