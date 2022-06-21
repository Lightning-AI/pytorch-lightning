import time

import requests
from requests.adapters import HTTPAdapter
from rich.progress import BarColumn, Progress, TextColumn
from urllib3.util.retry import Retry


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
    def upload_s3_data(url: str, data: bytes, retries: int, disconnect_retry_wait_seconds: int) -> str:
        """Send data to s3 url.

        Parameters
        ----------
        url: str
            S3 url string to send data to
        data: bytes
             Bytes of data to send to S3
        retries: int
            Amount of retries
        disconnect_retry_wait_seconds: int
            Amount of seconds between disconnect retry

        Returns
        -------
        str
            ETag from response
        """
        disconnect_retries = retries
        while disconnect_retries > 0:
            try:
                retries = Retry(total=10)
                with requests.Session() as s:
                    s.mount("https://", HTTPAdapter(max_retries=retries))
                    response = s.put(url, data=data)
                    if "ETag" not in response.headers:
                        raise ValueError(f"Unexpected response from S3, response: {response.content}")
                    return response.headers["ETag"]
            except BrokenPipeError:
                time.sleep(disconnect_retry_wait_seconds)
                disconnect_retries -= 1

        raise ValueError("Unable to upload file after multiple attempts")

    def upload(self) -> None:
        """Upload files from source dir into target path in S3."""
        task_id = self.progress.add_task("upload", filename=self.name, total=self.total_size)
        self.progress.start()
        try:
            with open(self.source_file, "rb") as f:
                data = f.read()
            self.upload_s3_data(self.presigned_url, data, self.retries, self.disconnect_retry_wait_seconds)
            self.progress.update(task_id, advance=len(data))
        finally:
            self.progress.stop()
