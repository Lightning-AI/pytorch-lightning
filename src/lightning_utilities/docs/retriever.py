# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import logging
import os
import re
from typing import List, Tuple

import requests


def _download_file(file_url: str, folder: str) -> str:
    """Download a file from URL to a particular folder."""
    fname = os.path.basename(file_url)
    file_path = os.path.join(folder, fname)
    if os.path.isfile(file_path):
        logging.warning(f'given file "{file_path}" already exists and will be overwritten with {file_url}')
    # see: https://stackoverflow.com/a/34957875
    rq = requests.get(file_url)
    with open(file_path, "wb") as outfile:
        outfile.write(rq.content)
    return fname


def _search_all_occurrences(list_files: List[str], pattern: str) -> List[str]:
    """Search for all occurrences of specific patter in a collection of files.

    Args:
        list_files: list of files to be scanned
        pattern: pattern for search, reg. expression
    """
    collected = []
    for file_path in list_files:
        with open(file_path, encoding="UTF-8") as fo:
            body = fo.read()
        found = re.findall(pattern, body)
        collected += found
    return collected


def _replace_remote_with_local(
    file_path: str, docs_folder: str, pairs_url_path: List[Tuple[str, str]], base_depth: int = 2
) -> None:
    """Replace all URL with local files in a given file.

    Args:
        file_path: file for replacement
        docs_folder: the location of docs related to the project root
        pairs_url_path: pairs of URL and local file path to be swapped
    """
    # drop the default/global path to the docs
    relt_path = os.path.dirname(file_path).replace(docs_folder, "")
    # filter the path starting with / as not empty folder names
    depth = len([p for p in relt_path.split(os.path.sep) if p])
    with open(file_path, encoding="UTF-8") as fo:
        body = fo.read()
    for url, fpath in pairs_url_path:
        if depth:
            path_up = [".."] * depth
            fpath = os.path.join(*path_up, fpath)
        body = body.replace(url, fpath)
    with open(file_path, "w", encoding="UTF-8") as fw:
        fw.write(body)


def fetch_external_assets(
    docs_folder: str = "docs/source",
    assets_folder: str = "fetched-s3-assets",
    file_pattern: str = "*.rst",
    retrieve_pattern: str = r"https?://[-a-zA-Z0-9_]+\.s3\.[-a-zA-Z0-9()_\\+.\\/=]+",
) -> None:
    """Search all URL in docs, download these files locally and replace online with local version.

    Args:
        docs_folder: the location of docs related to the project root
        assets_folder: a folder inside ``docs_folder`` to be created and saving online assets
        file_pattern: what kind of files shall be scanned
        retrieve_pattern: patter for reg. expression to search URL/S3 resources
    """
    list_files = glob.glob(os.path.join(docs_folder, "**", file_pattern), recursive=True)
    if not list_files:
        logging.warning(f'no files were listed in folder "{docs_folder}" and pattern "{file_pattern}"')
        return

    urls = _search_all_occurrences(list_files, pattern=retrieve_pattern)
    if not urls:
        logging.info(f"no resources/assets were match in {docs_folder} for {retrieve_pattern}")
        return
    target_folder = os.path.join(docs_folder, assets_folder)
    os.makedirs(target_folder, exist_ok=True)
    pairs_url_file = []
    for i, url in enumerate(set(urls)):
        logging.info(f" >> downloading ({i}/{len(urls)}): {url}")
        fname = _download_file(url, target_folder)
        pairs_url_file.append((url, os.path.join(assets_folder, fname)))

    for fpath in list_files:
        _replace_remote_with_local(fpath, docs_folder, pairs_url_file)
