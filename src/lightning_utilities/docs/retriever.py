# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import logging
import os
import re

import requests


def _download_file(file_url: str, folder: str) -> str:
    """Download a file from a URL into the given folder.

    If a file with the same name already exists, it will be overwritten.
    Returns the basename of the downloaded file. Network-related exceptions from
    ``requests.get`` (e.g., timeouts or connection errors) may propagate to the caller.

    """
    fname = os.path.basename(file_url)
    file_path = os.path.join(folder, fname)
    if os.path.isfile(file_path):
        logging.warning(f'given file "{file_path}" already exists and will be overwritten with {file_url}')
    # see: https://stackoverflow.com/a/34957875
    rq = requests.get(file_url, timeout=10)
    with open(file_path, "wb") as outfile:
        outfile.write(rq.content)
    return fname


def _search_all_occurrences(list_files: list[str], pattern: str) -> list[str]:
    """Search for all occurrences of a regular-expression pattern across files.

    Args:
        list_files: The list of file paths to scan.
        pattern: A regular-expression pattern to search for in each file.

    Returns:
        A list with all matches found across the provided files (order preserved per file).

    """
    collected = []
    for file_path in list_files:
        with open(file_path, encoding="UTF-8") as fopem:
            body = fopem.read()
        found = re.findall(pattern, body)
        collected += found
    return collected


def _replace_remote_with_local(file_path: str, docs_folder: str, pairs_url_path: list[tuple[str, str]]) -> None:
    """Replace all matching remote URLs with local file paths in a given file.

    Args:
        file_path: The file in which replacements should be performed.
        docs_folder: The documentation root folder (used to compute relative paths).
        pairs_url_path: Pairs of (remote_url, local_relative_path) to replace.

    """
    # drop the default/global path to the docs
    relt_path = os.path.dirname(file_path).replace(docs_folder, "")
    # filter the path starting with / as not empty folder names
    depth = len([p for p in relt_path.split(os.path.sep) if p])
    with open(file_path, encoding="UTF-8") as fopen:
        body = fopen.read()
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
    """Find S3 (or HTTP) asset URLs in docs, download them locally, and rewrite references to local paths.

    Args:
        docs_folder: The documentation root relative to the project.
        assets_folder: Subfolder inside ``docs_folder`` used to store downloaded assets (created if missing).
        file_pattern: Glob pattern of files to scan.
        retrieve_pattern: Regular-expression pattern used to find remote asset URLs.

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
