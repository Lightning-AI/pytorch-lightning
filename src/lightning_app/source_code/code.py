import os
from pathlib import Path

import requests

from lightning_app.source_code import LocalSourceCodeDir
from lightning_app.source_code.uploader import FileUploader


class CustomFileUploader(FileUploader):
    def _upload_data(self, s: requests.Session, url: str, data: bytes):
        resp = s.put(url, files={"uploaded_file": data})
        assert resp.status_code == 200


class CustomLocalSourceCodeDir(LocalSourceCodeDir):
    def upload(self, url: str) -> None:
        """Uploads package to URL, usually pre-signed URL.

        Notes
        -----
        Since we do not use multipart uploads here, we cannot upload any
        packaged repository files which have a size > 2GB.

        This limitation should be removed during the datastore upload redesign
        """
        if self.package_path.stat().st_size > 2e9:
            raise OSError(
                "cannot upload directory code whose total fize size is greater than 2GB (2e9 bytes)"
            ) from None

        uploader = CustomFileUploader(
            presigned_url=url,
            source_file=str(self.package_path),
            name=self.package_path.name,
            total_size=self.package_path.stat().st_size,
        )
        uploader.upload()


def upload_code(path: Path = Path(os.getcwd()).resolve()) -> None:
    repo = CustomLocalSourceCodeDir(path=path)
    repo.package()
    repo.upload(url=f"{os.getenv('LIGHTNING_APP_STATE_URL')}/api/v1/upload_file/code")
