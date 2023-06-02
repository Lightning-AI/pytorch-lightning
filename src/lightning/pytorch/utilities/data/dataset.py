import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import Dataset as TorchDataset

from lightning.pytorch.utilities.data.fileio import OpenCloudFileObj
from lightning.pytorch.utilities.data.get_index import get_index


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
    os.environ["AWS_SECRET_KEY"] = credentials.secret_key
    os.environ["AWS_SESSION_TOKEN"] = credentials.token

    return credentials


class LightningDataset(TorchDataset):
    """Dataset wrapper for optimized dataloading.

    Arguments:

        data_source: path of data directory.

        path_to_index_file: path to index file that lists all file contents of the data_source.
    """

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


class TempCreds:
    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key


class S3LightningDataset(LightningDataset, ABC):
    """LightningDataset for S3 buckets.

    Arguments:

        data_source: path of data directory.

        path_to_index_file: path to index file that lists all file contents of the data_source.
    """

    def __init__(self, data_source: str, path_to_index_file: Optional[str] = None):
        super().__init__(data_source=data_source, path_to_index_file=path_to_index_file)

        self.files = self.get_index()

        if os.getenv("AWS_ACCESS_KEY") and os.getenv("AWS_SECRET_KEY"):
            self.credentials = TempCreds(access_key=os.getenv("AWS_ACCESS_KEY"), secret_key=os.getenv("AWS_SECRET_KEY"))
        else:
            self.credentials = get_aws_credentials()

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
                "Unable to locate credentials. Make sure you have set the following environment variables: \nAWS_ACCESS_KEY\nAWS_SECRET_KEY"
            )
            raise ValueError(exc)
        except Exception as exc:
            raise ValueError(exc)

    @abstractmethod
    def load_sample(self, file_path: str, stream: OpenCloudFileObj) -> Any:
        pass

    def __len__(self) -> int:
        return len(self.files)
