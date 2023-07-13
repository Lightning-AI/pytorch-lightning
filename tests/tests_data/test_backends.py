import os
from typing import Mapping
from unittest import mock

import botocore.credentials
import botocore.utils

from lightning.data.backends import LocalDatasetBackend, S3DatasetBackend


def test_s3_dataset_backend_credentials_env_vars():
    os.environ["AWS_ACCESS_KEY"] = "123"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "abc"

    assert S3DatasetBackend().credentials() == {"access_key": "123", "secret_key": "abc"}
    os.environ.pop("AWS_ACCESS_KEY")
    os.environ.pop("AWS_SECRET_ACCESS_KEY")


def test_s3_dataset_backend_credentials_iam():
    botocore.credentials.InstanceMetadataProvider = mock.Mock()
    botocore.utils.InstanceMetadataFetcher = mock.Mock()

    credentials = S3DatasetBackend().credentials()
    assert isinstance(credentials, Mapping)
    assert all(key in credentials for key in ("token", "access_key", "secret_key"))


def test_local_dataset_backend_credentials():
    assert LocalDatasetBackend().credentials() == {}
