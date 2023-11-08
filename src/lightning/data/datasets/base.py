from typing import Any, Literal

from torch.utils.data import Dataset as TorchDataset

from lightning.data.backends import LocalDatasetBackend, S3DatasetBackend, _DatasetBackend
from lightning.data.fileio import OpenCloudFileObj


class _Dataset(TorchDataset):
    """Base dataset class for streaming data from a cloud storage.

    Args:
        backend: storage location of the data_source. current options are "s3" or "local"

    """

    def __init__(self, backend: Literal["local", "s3"] = "local"):
        self.backend = self._init_backend(backend=backend)

        assert isinstance(self.backend, _DatasetBackend)

    def _init_backend(self, backend: str) -> _DatasetBackend:
        """Picks the correct backend handler."""
        if backend == "s3":
            return S3DatasetBackend()
        if backend == "local":
            return LocalDatasetBackend()
        raise ValueError(f"Unsupported backend {backend}")

    def open(self, file: str, mode: str = "r", kwargs_for_open: Any = {}, **kwargs: Any) -> OpenCloudFileObj:
        """Opens a stream for the given file.

        Returns:
            A stream object of the file.

        """
        return OpenCloudFileObj(
            path=file, mode=mode, kwargs_for_open={**self.backend.credentials(), **kwargs_for_open}, **kwargs
        )
