import os
import shutil
import tarfile


def clean_tarfile(file_path: str, mode):
    if os.path.exists(file_path):
        with tarfile.open(file_path, mode=mode) as tar_ref:
            for member in tar_ref.getmembers():
                p = member.path
                if p != "." and os.path.exists(p):
                    if os.path.isfile(p):
                        os.remove(p)
                    else:
                        shutil.rmtree(p)
        os.remove(file_path)


def extract_tarfile(file_path: str, extract_path: str, mode: str):
    if os.path.exists(file_path):
        with tarfile.open(file_path, mode=mode) as tar_ref:
            for member in tar_ref.getmembers():
                try:
                    tar_ref.extract(member, path=extract_path, set_attrs=False)
                except PermissionError:
                    raise PermissionError(f"Could not extract tar file {file_path}")
