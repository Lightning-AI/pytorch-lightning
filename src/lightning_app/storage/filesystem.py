import os
import shutil
from pathlib import Path
from typing import Callable, List

from fsspec.implementations.local import LocalFileSystem

from lightning_app.storage.copier import _copy_files
from lightning_app.storage.path import _filesystem, _shared_storage_path


def _get_files(fs, src: Path, dst: Path, overwrite: bool = True):
    dst = dst.resolve()
    if fs.isdir(src):
        if isinstance(fs, LocalFileSystem):
            dst = dst.resolve()
            if fs.exists(dst):
                if overwrite:
                    fs.rm(str(dst), recursive=True)
                else:
                    raise FileExistsError(f"The file {dst} was found. Add get(..., overwrite=True) to replace it.")

            shutil.copytree(src, dst)
        else:
            glob = f"{str(src)}/**"
            fs.get(glob, str(dst), recursive=False)
    else:
        fs.get(str(src), str(dst), recursive=False)


class FileSystem:

    """This filesystem enables to easily move files from and to the shared storage."""

    def __init__(self) -> None:
        self._fs = _filesystem()
        self._root = str(_shared_storage_path())

    def put(self, src_path: str, dst_path: str, put_fn: Callable = _copy_files) -> None:
        """This method enables to put a file to the shared storage in a blocking fashion.

        Arguments:
            src_path: The path to your files locally
            dst_path: The path to your files transfered in the shared storage.
            put_fn: The method to use to put files in the shared storage.
        """
        if not os.path.exists(Path(src_path).resolve()):
            raise FileExistsError(f"The provided path {src_path} doesn't exist")

        if not dst_path.startswith("/"):
            raise Exception(f"The provided destination {dst_path} needs to start with `/`.")

        if dst_path == "/":
            dst_path = os.path.join(self._root, os.path.basename(src_path))
        else:
            dst_path = os.path.join(self._root, dst_path[1:])

        src = Path(src_path).resolve()
        dst = Path(dst_path).resolve()

        return put_fn(src, dst, fs=self._fs)

    def get(self, src_path: str, dst_path: str, overwrite: bool = True, get_fn: Callable = _get_files) -> None:
        """This method enables to get files from the shared storage in a blocking fashion.

        Arguments:
            src_path: The path to your files in the shared storage
            dst_path: The path to your files transfered locally
            get_fn: The method to use to put files in the shared storage.
        """
        if not src_path.startswith("/"):
            raise Exception(f"The provided destination {src_path} needs to start with `/`.")

        src = Path(os.path.join(self._root, src_path[1:])).resolve()
        dst = Path(dst_path).resolve()

        return get_fn(fs=self._fs, src=src, dst=dst, overwrite=overwrite)

    def listdir(self, path: str) -> List[str]:
        """This method enables to list files from the shared storage in a blocking fashion.

        Arguments:
            path: The path to files to list.
        """
        if not path.startswith("/"):
            raise Exception(f"The provided destination {path} needs to start with `/`.")

        shared_path = Path(os.path.join(self._root, path[1:])).resolve()

        if not self._fs.exists(shared_path):
            raise RuntimeError(f"The provided path {shared_path} doesn't exist.")

        # Invalidate cache before running ls in case new directories have been added
        # TODO: Re-evaluate this - may lead to performance issues
        self._fs.invalidate_cache()

        paths = self._fs.ls(shared_path)
        if not paths:
            return paths

        return sorted([path.replace(self._root + os.sep, "") for path in paths if not path.endswith("info.txt")])

    def walk(self, path: str) -> List[str]:
        """This method enables to list files from the shared storage in a blocking fashion.

        Arguments:
            path: The path to files to list.
        """
        if not path.startswith("/"):
            raise Exception(f"The provided destination {path} needs to start with `/`.")

        shared_path = Path(os.path.join(self._root, path[1:])).resolve()

        if not self._fs.exists(shared_path):
            raise RuntimeError(f"The provided path {shared_path} doesn't exist.")

        # Invalidate cache before running ls in case new directories have been added
        # TODO: Re-evaluate this - may lead to performance issues
        self._fs.invalidate_cache()

        paths = self._fs.ls(shared_path)
        if not paths:
            return paths

        out = []

        for shared_path in paths:
            path = str(shared_path).replace(self._root, "")
            if self._fs.isdir(shared_path):
                out.extend(self.walk(path))
            else:
                if path.endswith("info.txt"):
                    continue
                out.append(path[1:])
        return sorted(out)

    def rm(self, path) -> None:
        if not path.startswith("/"):
            raise Exception(f"The provided destination {path} needs to start with `/`.")

        delete_path = Path(os.path.join(self._root, path[1:])).resolve()

        if self._fs.exists(str(delete_path)):
            if self._fs.isdir(str(delete_path)):
                self._fs.rmdir(str(delete_path))
            else:
                self._fs.rm(str(delete_path))
        else:
            raise Exception(f"The file path {path} doesn't exist.")

    def isfile(self, path: str) -> bool:
        if not path.startswith("/"):
            raise Exception(f"The provided destination {path} needs to start with `/`.")

        path = Path(os.path.join(self._root, path[1:])).resolve()
        return self._fs.isfile(path)

    def isdir(self, path: str) -> bool:
        if not path.startswith("/"):
            raise Exception(f"The provided destination {path} needs to start with `/`.")

        path = Path(os.path.join(self._root, path[1:])).resolve()
        return self._fs.isdir(path)
