import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict

from torch.utils.data import Dataset as TorchDataset

from lightning.pytorch.utilities.data.fileio import OpenCloudFileObj
from lightning.pytorch.utilities.data.get_index import get_index


def get_aws_credentials():
    from botocore.credentials import InstanceMetadataProvider
    from botocore.utils import InstanceMetadataFetcher

    provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=1000, num_attempts=2))

    credentials = provider.load()

    os.environ["AWS_ACCESS_KEY"] = credentials.access_key
    os.environ["AWS_SECRET_KEY"] = credentials.secret_key
    os.environ["AWS_SESSION_TOKEN"] = credentials.token

    return credentials


class LightningDataset(TorchDataset):
    def __init__(self, data_source: str, path_to_index_file: Optional[str] = None):
        super().__init__()
        self.data_source = data_source

        if path_to_index_file is None:
            tmpdir = tempfile.mkdtemp()
            path_to_index_file = os.path.join(tmpdir, "index.txt")

        path_to_index_file = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_index_file)))

        os.makedirs(os.path.dirname(path_to_index_file), exist_ok=True)
        self.index_file = path_to_index_file

    def get_index(self) -> Tuple[str, ...]:
        if not os.path.isfile(self.index_file):
            get_index(self.data_source, self.index_file)

        with open(self.index_file) as f:
            index = f.readlines()
        return (line.strip("\n") for line in index)

    @staticmethod
    def open(file: str, mode: str = "r", kwargs_for_open: Optional[Dict] = None, **kwargs):
        return OpenCloudFileObj(file, mode=mode, kwargs_for_open=kwargs_for_open, **kwargs)


class S3LightningDataset(LightningDataset, ABC):
    def __init__(self, data_source: str, path_to_index_file: Optional[str] = None):
        super().__init__(data_source=data_source, path_to_index_file=path_to_index_file)

        self.files = self.get_index()
        self.credentials = get_aws_credentials()

    def __getitem__(self, idx: int) -> Any:
        file_path = self.files[idx]

        with self.open(
            file_path,
            "rb",
            key=self.credentials.access_key,
            secret=self.credentials.secret_key,
            token=self.credentials.token,
        ) as stream:
            return self.load_sample(file_path, stream)

    @abstractmethod
    def load_sample(self, file_path: str, stream: OpenCloudFileObj) -> Any:
        pass

    def __len__(self) -> int:
        return len(self.files)
