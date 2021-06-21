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
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _load_requirements(path_dir: str, file_name: str = 'requirements.txt', comment_char: str = '#') -> List[str]:
    """Load requirements from a file

    >>> _load_requirements(_PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
        if ln.startswith('http') or '@http' in ln:
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def _load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as decribtion

    >>> _load_readme_description(_PROJECT_ROOT, "", "")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<div align="center">...'
    """
    path_readme = os.path.join(path_dir, "README.md")
    text = open(path_readme, encoding="utf-8").read()

    # drop images from readme
    text = text.replace('![PT to PL](docs/source/_static/images/general/pl_quick_start_full_compressed.gif)', '')

    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_static/images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(homepage, "raw", version)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we reger some other sources with particular path to docs
    text = text.replace("docs/source/_static/", f"{os.path.join(github_source_url, 'docs/source/_static/')}")

    # readthedocs badge
    text = text.replace('badge/?version=stable', f'badge/?version={version}')
    text = text.replace('pytorch-lightning.readthedocs.io/en/stable/', f'pytorch-lightning.readthedocs.io/en/{version}')
    # codecov badge
    text = text.replace('/branch/master/graph/badge.svg', f'/release/{version}/graph/badge.svg')
    # replace github badges for release ones
    text = text.replace('badge.svg?branch=master&event=push', f'badge.svg?tag={version}')
    # Azure...
    text = text.replace('?branchName=master', f'?branchName=refs%2Ftags%2F{version}')
    text = re.sub(r'\?definitionId=\d+&branchName=master', f'?definitionId=2&branchName=refs%2Ftags%2F{version}', text)

    skip_begin = r'<!-- following section will be skipped from PyPI description -->'
    skip_end = r'<!-- end skipping PyPI description -->'
    # todo: wrap content as commented description
    text = re.sub(rf"{skip_begin}.+?{skip_end}", '<!--  -->', text, flags=re.IGNORECASE + re.DOTALL)

    # # https://github.com/Borda/pytorch-lightning/releases/download/1.1.0a6/codecov_badge.png
    # github_release_url = os.path.join(homepage, "releases", "download", version)
    # # download badge and replace url with local file
    # text = _parse_for_badge(text, github_release_url)
    return text
