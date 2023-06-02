import os
import random
import time
from typing import Dict, Optional


def path_to_url(path: str, bucket_name: str, bucket_root_path: str = "/") -> str:
    if not path.startswith(bucket_root_path):
        raise ValueError(f"Cannot create a path from {path} relative to {bucket_root_path}")
    return f"s3://{bucket_name}/{os.path.relpath(path, bucket_root_path)}"


def open_single_file(path_or_url, mode: str = "r", kwargs_for_open: Optional[Dict] = None, **kwargs):
    """Streams the given file.

    Returns:
        The opened file stream.
    """
    from torchdata.datapipes.iter import FSSpecFileOpener, IterableWrapper

    datapipe = IterableWrapper([path_or_url])

    # iterable of length 1, still better than manually instantiating iterator and calling next
    for _, stream in FSSpecFileOpener(datapipe, mode=mode, kwargs_for_open=kwargs_for_open, **kwargs):
        return stream
    return None


def open_single_file_with_retry(path_or_url, mode: str = "r", kwargs_for_open: Optional[Dict] = None, **kwargs):
    """Streams the given file with a retry mechanism in case of high batch_size (>128) parallel opens.

    Returns:
        The opened file stream.
    """
    from botocore.exceptions import NoCredentialsError
    from torchdata.datapipes.iter import FSSpecFileOpener, IterableWrapper

    datapipe = IterableWrapper([path_or_url], **kwargs)

    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            for _, stream in FSSpecFileOpener(datapipe, mode=mode, kwargs_for_open=kwargs_for_open, **kwargs):
                return stream
        except NoCredentialsError:
            print(f"Could not locate credentials, retrying: attempt {attempt}/{num_attempts}")

        time.sleep(15 * (random.random() + 0.5))
    raise RuntimeError()


# Necessary to support both a context manager and a call
class OpenCloudFileObj:
    """File object wrapper that streams files on open.

    Arguments:

        path: string containg the path of the file to be opened.

        mode: An optional string that specifies the mode in which the file is opened (``"r"`` by default).

        kwargs_for_open: Optional Dict to specify kwargs for opening files (``fs.open()``).
    """

    def __init__(self, path: str, mode: str = "r", kwargs_for_open: Optional[Dict] = None, **kwargs):
        from torchdata.datapipes.utils import StreamWrapper

        self._path = path
        self._stream: Optional["StreamWrapper"] = None
        self._mode = mode
        self._kwargs_for_open = kwargs_for_open
        self._kwargs = kwargs

    def __enter__(self):
        return self._conditionally_open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.close()

    def _conditionally_open(self):
        if self._stream is None:
            self._stream = open_single_file(
                self._path, mode=self._mode, kwargs_for_open=self._kwargs_for_open, **self._kwargs
            )

        return self._stream

    def _conditionally_close(self):
        if self._stream is not None:
            self._stream.close()

    def __call__(self):
        return self._conditionally_open()

    def __getattr__(self, attr: str):
        return getattr(self._conditionally_open(), attr)
