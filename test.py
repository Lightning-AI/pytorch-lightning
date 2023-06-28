from botocore.credentials import InstanceMetadataProvider
from botocore.utils import InstanceMetadataFetcher
from s3fs import S3FileSystem

from lightning.app.storage.path import _shared_storage_path

provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=1000, num_attempts=2))

credentials = provider.load()

print(credentials)

fs = S3FileSystem(key=credentials.access_key, secret=credentials.secret_key, token=credentials.token)

if not fs.exists(_shared_storage_path()):
    raise RuntimeError(f"shared filesystem {_shared_storage_path()} does not exist")
