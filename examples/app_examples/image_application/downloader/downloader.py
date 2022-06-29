import os
import tarfile
import zipfile
from typing import Optional

import requests

import lightning as L
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState

DEFAULT_DOWNLOAD_URL = "https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip"


class DownloaderWork(L.LightningWork):
    def __init__(
        self,
        download_url: Optional[str] = None,
        target_dir: str = "data/",
        chunk_size: int = 1024,
    ):
        """The DownloaderWork is responsible to download the data from a given url locally.

        Arguments:
            download_url: Download url to fetch the data from.
            target_dir: Directory where the data are being downloaded to.
            chunk_size: The size of each chunk of data to download on each request.
        """
        super().__init__()
        self.download_url: Optional[str] = download_url or DEFAULT_DOWNLOAD_URL
        self.file_size: Optional[int] = None
        self.target_dir = target_dir
        self.chunk_size = chunk_size
        self.progress: float = 0
        self.has_completed = False
        self.destination_dir = None

    def run(self):
        path = os.path.join(L._PROJECT_ROOT, self.target_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        local_filename = os.path.join(path, self.download_url.split("/")[-1])
        self.destination_dir = local_filename.split(".")[0]

        if os.path.exists(self.destination_dir):
            self.has_completed = True
            self.progress = 1.0
            return

        if os.path.exists(local_filename):
            os.remove(local_filename)

        assert self.download_url
        r = requests.get(self.download_url, stream=True, verify=False)
        self.file_size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else 0

        if not os.path.exists(local_filename):
            with open(local_filename, "wb") as fp:
                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    fp.write(chunk)
                    extra = float(self.chunk_size) / self.file_size
                    self.progress = min(self.progress + extra, 1.0)

        def extract_tarfile(file_path: str, extract_path: str, mode: str):
            if os.path.exists(file_path):
                with tarfile.open(file_path, mode=mode) as tar_ref:
                    for member in tar_ref.getmembers():
                        try:
                            tar_ref.extract(member, path=extract_path, set_attrs=False)
                        except PermissionError:
                            raise PermissionError(f"Could not extract tar file {file_path}")

        if ".zip" in local_filename:
            if os.path.exists(local_filename):
                with zipfile.ZipFile(local_filename, "r") as zip_ref:
                    zip_ref.extractall(path)
        elif local_filename.endswith(".tar.gz") or local_filename.endswith(".tgz"):
            extract_tarfile(local_filename, path, "r:gz")
        elif local_filename.endswith(".tar.bz2") or local_filename.endswith(".tbz"):
            extract_tarfile(local_filename, path, "r:bz2")

        self.has_completed = True


class Downloader(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = DownloaderWork()
        self.should_download = False

    @property
    def destination_dir(self):
        return self.work.destination_dir

    @property
    def has_completed(self):
        return self.work.has_completed

    @property
    def has_already_downloaded(self) -> bool:
        download_url = self.work.download_url
        if not download_url:
            return False

        path = os.path.join(L._PROJECT_ROOT, self.work.target_dir)
        local_filename = os.path.join(path, download_url.split("/")[-1])
        destination_dir = local_filename.split(".")[0]
        if os.path.exists(destination_dir):
            return True
        return False

    def run(self):
        if (self.work.download_url and self.should_download) or self.has_already_downloaded:
            self.work.run()

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):
    """This method would be running StreamLit within its own process.

    Arguments:
        state: Connection to this flow state. Enable changing the state from the UI.
    """
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=2000, limit=None, key="refresh")

    download_url = st.text_input("download_url", DEFAULT_DOWNLOAD_URL)
    should_download = st.button("Should download ?")

    if should_download:
        state.should_download = should_download

    state.work.download_url = download_url

    st.write(state.work.progress)

    st.progress(state.work.progress)

    if state.work.has_completed:

        st.write(f"The data are downloaded at {state.work.destination_dir}")
