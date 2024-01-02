import os
import os.path
import tarfile
import zipfile

import requests


def download_data(url: str, path: str = "data/", verbose: bool = False) -> None:
    """Download file with progressbar.

    # Code taken from: https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603
    # __author__  = "github.com/ruxi"
    # __license__ = "MIT"

    Usage:
        download_file('http://web4host.net/5MB.zip')

    """
    if url == "NEED_TO_BE_CREATED":
        raise NotImplementedError

    if not os.path.exists(path):
        os.makedirs(path)
    local_filename = os.path.join(path, url.split("/")[-1])
    r = requests.get(url, stream=True, verify=False)
    file_size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else 0
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)
    if verbose:
        print({"file_size": file_size})
        print({"num_bars": num_bars})

    if not os.path.exists(local_filename):
        with open(local_filename, "wb") as fp:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fp.write(chunk)  # type: ignore

    def extract_tarfile(file_path: str, extract_path: str, mode: str):
        if os.path.exists(file_path):
            with tarfile.open(file_path, mode=mode) as tar_ref:
                for member in tar_ref.getmembers():
                    try:
                        tar_ref.extract(member, path=extract_path, set_attrs=False)
                    except PermissionError:
                        raise PermissionError(f"Could not extract tar file {file_path}")

    if ".zip" in local_filename:
        if os.path.exists(local_filename):
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(path)  # noqa: S202
    elif local_filename.endswith(".tar.gz") or local_filename.endswith(".tgz"):
        extract_tarfile(local_filename, path, "r:gz")
    elif local_filename.endswith(".tar.bz2") or local_filename.endswith(".tbz"):
        extract_tarfile(local_filename, path, "r:bz2")
