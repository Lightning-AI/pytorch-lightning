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

from urllib.error import URLError
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import warnings

from pytorch_lightning import __homepage__, __version__, PROJECT_ROOT

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
    ['numpy>=...', 'torch>=...', ...]
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


def _parse_for_badge(text, badge_names: list = _DEFAULT_BADGES):
    for line in text.split('\n'):
        badge_name = re.search(r'^\[!\[(.*?)]', line)

        # check for the badge name
        if badge_name is not None:
            badge_name = badge_name.group(1)

            # check if valid name
            if badge_name in badge_names:

                search_string = fr'\[\!\[{badge_name}]\((.*?)\)]'
                badge_url = re.search(search_string, line)
                # check for badge url
                if badge_url is not None:
                    badge_url = badge_url.group(1)

                # download badge
                saved_badge_name = _download_badges(badge_url, badge_name)

                # replace url with local file path
                replace_string = f'[![{badge_name}]({saved_badge_name})]'
                text = re.sub(search_string, replace_string, text)

    return text


def _download_badges(url_badge, badge_name):

    base_path = 'docs/source/_images/badges'
    os.makedirs(base_path, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/svg,*/*;q=0.8',
    }

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
            return save_path + '.svg'


def _load_long_description(path_dir):
    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    url = os.path.join(__homepage__, 'raw', __version__, 'docs')
    path_readme = os.path.join(path_dir, 'README.md')
    text = open(path_readme, encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace('.svg', '.png')
    # download badge and replace url with local file
    text = _parse_for_badge(text)
    return text
