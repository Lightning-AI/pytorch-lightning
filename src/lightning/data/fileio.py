import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    try:
        from torchdata.datapipes.utils import StreamWrapper
    except ImportError:
        StreamWrapper = object


def is_url(path: str) -> bool:
    return path.startswith("s3://")


def is_path(path: str) -> bool:
    return not is_url(path) and path.startswith("/")


def path_to_url(path: str, bucket_name: str, bucket_root_path: str = "/") -> str:
    """Gets full S3 path given bucket info.

    Returns:
        Full S3 url path

    """
    if not path.startswith(bucket_root_path):
        raise ValueError(f"Cannot create a path from {path} relative to {bucket_root_path}")

    rel_path = os.path.relpath(path, bucket_root_path).replace("\\", "/")
    return f"s3://{bucket_name}/{rel_path}"


def open_single_file(
    path_or_url: str, mode: str = "r", kwargs_for_open: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> "StreamWrapper":
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


def open_single_file_with_retry(
    path_or_url: str, mode: str = "r", kwargs_for_open: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> "StreamWrapper":
    """Streams the given file with a retry mechanism in case of high batch_size (>128) parallel opens.

    Returns:
        The opened file stream.

    """
    from torchdata.datapipes.iter import FSSpecFileOpener, IterableWrapper

    datapipe = IterableWrapper([path_or_url], **kwargs)

    num_attempts = 5

    for _, stream in FSSpecFileOpener(datapipe, mode=mode, kwargs_for_open=kwargs_for_open, **kwargs):
        curr_attempt = 0
        while curr_attempt < num_attempts:
            try:
                return stream
            except Exception:
                curr_attempt += 1

    raise RuntimeError(f"Could not open {path_or_url}")


# Necessary to support both a context manager and a call
class OpenCloudFileObj:
    """File object wrapper that streams files on open.

    Arguments:

        path: string containg the path of the file to be opened.

        mode: An optional string that specifies the mode in which the file is opened (``"r"`` by default).

        kwargs_for_open: Optional Dict to specify kwargs for opening files (``fs.open()``).

    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        kwargs_for_open: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        from torchdata.datapipes.utils import StreamWrapper

        self._path = path
        self._stream: Optional[StreamWrapper] = None
        self._mode = mode
        self._kwargs_for_open = kwargs_for_open
        self._kwargs = kwargs

    def __enter__(self) -> "StreamWrapper":
        return self._conditionally_open()

    def __exit__(self, exc_type: str, exc_val: str, exc_tb: str) -> None:
        if self._stream is not None:
            self._stream.close()

    def _conditionally_open(self) -> "StreamWrapper":
        if self._stream is None:
            self._stream = open_single_file(
                self._path, mode=self._mode, kwargs_for_open=self._kwargs_for_open, **self._kwargs
            )

        return self._stream

    def _conditionally_close(self) -> None:
        if self._stream is not None:
            self._stream.close()

    def __call__(self) -> "StreamWrapper":
        return self._conditionally_open()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._conditionally_open(), attr)
