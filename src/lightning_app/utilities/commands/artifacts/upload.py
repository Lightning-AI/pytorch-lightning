import argparse
import os
from pathlib import Path

import requests
from rich.progress import BarColumn, Progress, TextColumn

from lightning_app.core.constants import APP_SERVER_HOST, APP_SERVER_PORT, UPLOAD_FILENAME_SEPARATOR
from lightning_app.utilities.commands import ClientCommand


class UploadArtifactsCommand(ClientCommand):

    """This command enables to upload artifacts to the Shared Drive of the App.

    Example::

        from lightning_app.storage import Drive

        drive = Drive("lit://uploaded_files")
        drive.list() # Returns all the uploaded files
    """

    def run(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--files", nargs="+", default=[], help="Provide a list of files.")
        hparams = parser.parse_args()

        app_url = os.getenv("LIGHTNING_APP_STATE_URL", f"{APP_SERVER_HOST}:{APP_SERVER_PORT}")
        upload_endpoint = f"{app_url}/api/v1/upload_file"

        progress = Progress(
            TextColumn("[bold blue]{task.description}", justify="left"),
            BarColumn(bar_width=None),
            "[self.progress.percentage]{task.percentage:>3.1f}%",
        )

        files = [Path(file).resolve().relative_to(os.getcwd()) for file in hparams.files]

        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"The file {file} wasn't found.")

        upload_files = []

        for file in files:
            if os.path.isfile(file):
                upload_files.append(file)
            else:
                upload_files.extend(
                    [
                        Path(os.path.join(root, sub_file)).resolve().relative_to(os.getcwd())
                        for root, _, sub_files in os.walk(file)
                        for sub_file in sub_files
                    ]
                )

        total = float(sum(f.stat().st_size for f in upload_files))

        task_id = progress.add_task("upload")
        progress.start()
        for upload_file in upload_files:
            try:
                with open(upload_file, "rb") as f:
                    data = f.read()
                    upload_file = str(upload_file).replace("/", UPLOAD_FILENAME_SEPARATOR)
                    requests.put(f"{upload_endpoint}/{upload_file}", files={"uploaded_file": data})
                    progress.update(task_id, advance=100 * float(len(data) / total))
            finally:
                pass
        progress.stop()


def empty_fn():
    return None


UPLOAD_ARTIFACT = {"upload artifacts": UploadArtifactsCommand(empty_fn)}
