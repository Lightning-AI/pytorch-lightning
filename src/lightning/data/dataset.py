import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional

from torch.utils.data import Dataset as TorchDataset

from lightning.data.backends import _DatasetBackend, LocalDatasetBackend, S3DatasetBackend
from lightning.data.dataset_index import get_index
from lightning.data.fileio import OpenCloudFileObj


class LightningDataset(TorchDataset, ABC):
    """Dataset wrapper for optimized dataloading.

    Arguments:

        data_source: path of data directory. ex. s3://mybucket/path

        backend: storage location of the data_source. current options are "s3" or "local"

        path_to_index_file: path to index file that lists all file contents of the data_source.
    """

    def __init__(self, data_source: str, backend: str = "local", path_to_index_file: Optional[str] = None):
        super().__init__()
        self.data_source = data_source

        if not path_to_index_file:
            tmpdir = tempfile.mkdtemp()
            path_to_index_file = os.path.join(tmpdir, "index.txt")

        self.index_file = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_index_file)))

        self.files = self.get_index()

        self.backend = self._init_backend(backend=backend)

        assert isinstance(self.backend, _DatasetBackend)

    def _init_backend(self, backend: str) -> _DatasetBackend:
        """Picks the correct backend handler."""
        if backend == "s3":
            return S3DatasetBackend()
        if backend == "local":
            return LocalDatasetBackend()
        raise ValueError(f"Unsupported backend {backend}")

    def get_index(self) -> Any:
        """Gets existing index or triggers an index generation if it doesn't exist for the provided data_source.

        Returns:
            The contents of the index file (all the file paths in the data_source)
        """
        if not os.path.isfile(self.index_file):
            get_index(self.data_source, self.index_file)

        with open(self.index_file) as f:
            index = f.readlines()
        return (line.strip("\n") for line in index)

    def open(self, file: str, mode: str = "r", kwargs_for_open: Any = {}, **kwargs: Any) -> OpenCloudFileObj:
        """Opens a stream for the given file.

        Returns:
            A stream object of the file.
        """
        return OpenCloudFileObj(
            path=file, mode=mode, kwargs_for_open={**self.backend.credentials(), **kwargs_for_open}, **kwargs
        )

    def __getitem__(self, idx: int) -> Any:
        """Get's item from the dataset at provided index.

        Returns:
            The loaded item
        """
        file_path = self.files[idx]

        try:
            with self.open(
                file_path,
                "rb",
            ) as stream:
                return self.load_sample(file_path, stream)
        except Exception as exc:
            self.backend.handle_error(exc)

    @abstractmethod
    def load_sample(self, file_path: str, stream: OpenCloudFileObj) -> Any:
        """Loads each sample in the dataset.

        Any data prep/cleaning logic goes here. For ex. image transformations, text cleaning, etc.
        """
        pass

    def __len__(self) -> int:
        return len(self.files)
