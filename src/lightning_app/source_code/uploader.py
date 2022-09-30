import sys
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from rich.progress import BarColumn, Progress, TextColumn
from urllib3.util.retry import Retry


class UploadThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_upload_exception = False

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.has_upload_exception = True
            raise e


class FileUploader:
    """This class uploads a source file with presigned url to S3.

    Attributes
    ----------
    source_file: str
        Source file to upload
    presigned_url: str
        Presigned urls dictionary, with key as part number and values as urls
    retries: int
        Amount of retries when requests encounter an error
    total_size: int
        Size of all files to upload
    name: str
        Name of this upload to display progress
    """

    workers: int = 8
    retries: int = 10000
    disconnect_retry_wait_seconds: int = 5

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        "[self.progress.percentage]{task.percentage:>3.1f}%",
    )

    def __init__(self, presigned_url: str, source_file: str, total_size: int, name: str):
        self.presigned_url = presigned_url
        self.source_file = source_file
        self.total_size = total_size
        self.name = name

    @staticmethod
    def upload_data(url: str, data: bytes):
        """Send data to url.

        Parameters
        ----------
        url: str
            url string to send data to
        data: bytes
             Bytes of data to send to url
        """
        resp = requests.put(url, data=data)
        if resp.status_code != 200:
            raise Exception(f"Error uploading data to {url}: {resp.text}")
        if "ETag" not in resp.headers:
            raise ValueError(f"Unexpected response from {url}, response: {resp.content}")

    def upload(self) -> None:
        """Upload files from source dir into target path in S3."""
        task_id = self.progress.add_task("upload", filename=self.name, total=self.total_size)
        self.progress.start()
        try:
            with open(self.source_file, "rb") as f:
                data = f.read()
            thread = UploadThread(daemon=True, target=self.upload_data, args=(self.presigned_url, data))  # noqa
            thread.start()
            while True:
                if not thread.is_alive():
                    break
                time.sleep(0.3)
            if thread.has_upload_exception:
                # if the thread had an exception, it's already redirected to stderr, we don't want to re-raise an
                # exception here. But exiting with a non-zero exit code
                sys.exit(1)
            self.progress.update(task_id, advance=len(data))
        finally:
            self.progress.stop()
            pass
