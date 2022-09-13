import os
import shutil
import tarfile


def clean_tarfile(file_path: str, mode: str) -> None:
    """This utility removes all files extracted from a tarfile."""

    if not os.path.exists(file_path):
        return None

    with tarfile.open(file_path, mode=mode) as tar_ref:
        for member in tar_ref.getmembers():
            p = member.path
            if p == "." or not os.path.exists(p):
                continue
            try:
                if os.path.isfile(p):
                    os.remove(p)
                else:
                    shutil.rmtree(p)
            except (FileNotFoundError, OSError, PermissionError):
                pass

    if os.path.exists(file_path):
        os.remove(file_path)


def extract_tarfile(file_path: str, extract_path: str, mode: str) -> None:
    """This utility extracts all files from a tarfile."""
    if not os.path.exists(file_path):
        return None

    with tarfile.open(file_path, mode=mode) as tar_ref:
        for member in tar_ref.getmembers():
            try:
                tar_ref.extract(member, path=extract_path, set_attrs=False)
            except PermissionError:
                raise PermissionError(f"Could not extract tar file {file_path}")
