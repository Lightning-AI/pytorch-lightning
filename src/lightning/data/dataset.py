import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import Dataset as TorchDataset

from lightning.data.dataset_index import get_index
from lightning.data.fileio import OpenCloudFileObj


def get_aws_credentials():
    """Gets AWS credentials from the current IAM role.

    Returns:
        credentials object to be used for file reading
    """
    from botocore.credentials import InstanceMetadataProvider
    from botocore.utils import InstanceMetadataFetcher

    provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=1000, num_attempts=2))

    credentials = provider.load()

    os.environ["AWS_ACCESS_KEY"] = credentials.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
    os.environ["AWS_SESSION_TOKEN"] = credentials.token

    return credentials


class CredsObj:
    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key


class LightningDataset(TorchDataset, ABC):
    """Dataset wrapper for optimized dataloading.

    Arguments:

        data_source: path of data directory.

        backend: current options are "s3" or "local"

        path_to_index_file: path to index file that lists all file contents of the data_source.
    """

    def __init__(self, data_source: str, backend: str = "s3", path_to_index_file: Optional[str] = None):
        super().__init__()
        self.data_source = data_source
        self.backend = backend

        if not path_to_index_file:
            tmpdir = tempfile.mkdtemp()
            path_to_index_file = os.path.join(tmpdir, "index.txt")

        self.index_file = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_index_file)))

        self.files = self.get_index()

        if os.getenv("AWS_ACCESS_KEY") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            self.credentials = CredsObj(
                access_key=os.getenv("AWS_ACCESS_KEY"), secret_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
        else:
            self.credentials = get_aws_credentials()

    def get_index(self) -> Tuple[str, ...]:
        """Gets existing index or triggers an index generation if it doesn't exist for the provided data_source.

        Returns:
            The contents of the index file (all the file paths in the data_source)
        """
        if not os.path.isfile(self.index_file):
            get_index(self.data_source, self.index_file)

        with open(self.index_file) as f:
            index = f.readlines()
        return (line.strip("\n") for line in index)

    @staticmethod
    def open(file: str, mode: str = "r", kwargs_for_open: Optional[Dict] = None, **kwargs):
        return OpenCloudFileObj(file, mode=mode, kwargs_for_open=kwargs_for_open, **kwargs)

    def __getitem__(self, idx: int) -> Any:
        from botocore.exceptions import NoCredentialsError

        file_path = self.files[idx]

        try:
            with self.open(
                file_path,
                "rb",
                key=self.credentials.access_key,
                secret=self.credentials.secret_key,
            ) as stream:
                return self.load_sample(file_path, stream)
        except NoCredentialsError as exc:
            print(
                "Unable to locate credentials. \
                Make sure you have set the following environment variables: \nAWS_ACCESS_KEY\\AWS_SECRET_ACCESS_KEY"
            )
            raise ValueError(exc)
        except Exception as exc:
            raise ValueError(exc)

    @abstractmethod
    def load_sample(self, file_path: str, stream: OpenCloudFileObj) -> Any:
        pass

    def __len__(self) -> int:
        return len(self.files)
