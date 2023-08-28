import json
import os
import tarfile
import uuid
import zipfile
from pathlib import Path

from lightning.app import LightningWork, LightningApp
from lightning.app.storage import Drive


class FileServer(LightningWork):
    def __init__(
        self,
        drive: Drive,
        base_dir: str = "file_server",
        chunk_size=10240,
        **kwargs
    ):
        """This component uploads, downloads files to your application.

        Arguments:
            drive: The drive can share data inside your application.
            base_dir: The local directory where the data will be stored.
            chunk_size: The quantity of bytes to download/upload at once.

        """
        super().__init__(
            cloud_build_config=BuildConfig(["flask, flask-cors"]),
            parallel=True,
            **kwargs,
        )
        # 1: Attach the arguments to the state.
        self.drive = drive
        self.base_dir = base_dir
        self.chunk_size = chunk_size

        # 2: Create a folder to store the data.
        os.makedirs(self.base_dir, exist_ok=True)

        # 3: Keep a reference to the uploaded filenames.
        self.uploaded_files = dict()

    def get_filepath(self, path: str) -> str:
        """Returns file path stored on the file server."""
        return os.path.join(self.base_dir, path)

    def get_random_filename(self) -> str:
        """Returns a random hash for the file name."""
        return uuid.uuid4().hex

    def upload_file(self, file):
        """Upload a file while tracking its progress."""
        # 1: Track metadata about the file
        filename = file.filename
        uploaded_file = self.get_random_filename()
        meta_file = uploaded_file + ".meta"
        self.uploaded_files[filename] = {
            "progress": (0, None), "done": False
        }

        # 2: Create a stream and write bytes of
        # the file to the disk under `uploaded_file` path.
        with open(self.get_filepath(uploaded_file), "wb") as out_file:
            content = file.read(self.chunk_size)
            while content:
                # 2.1 Write the file bytes
                size = out_file.write(content)

                # 2.2 Update the progress metadata
                self.uploaded_files[filename]["progress"] = (
                    self.uploaded_files[filename]["progress"][0] + size,
                    None,
                )
                # 4: Read next chunk of data
                content = file.read(self.chunk_size)

        # 3: Update metadata that the file has been uploaded.
        full_size = self.uploaded_files[filename]["progress"][0]
        self.drive.put(self.get_filepath(uploaded_file))
        self.uploaded_files[filename] = {
            "progress": (full_size, full_size),
            "done": True,
            "uploaded_file": uploaded_file,
        }

        # 4: Write down the metadata about the file to the disk
        meta = {
            "original_path": filename,
            "display_name": os.path.splitext(filename)[0],
            "size": full_size,
            "drive_path": uploaded_file,
        }
        with open(self.get_filepath(meta_file), "w") as f:
            json.dump(meta, f)

        # 5: Put the file to the drive.
        # It means other components can access get or list them.
        self.drive.put(self.get_filepath(meta_file))
        return meta

    def list_files(self, file_path: str):
        # 1: Get the local file path of the file server.
        file_path = self.get_filepath(file_path)

        # 2: If the file exists in the drive, transfer it locally.
        if not os.path.exists(file_path):
            self.drive.get(file_path)

        if os.path.isdir(file_path):
            result = set()
            for _, _, f in os.walk(file_path):
                for file in f:
                    if not file.endswith(".meta"):
                        for filename, meta in self.uploaded_files.items():
                            if meta["uploaded_file"] == file:
                                result.add(filename)
            return {"asset_names": [v for v in result]}

        # 3: If the filepath is a tar or zip file, list their contents
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, "r") as zf:
                result = zf.namelist()
        elif tarfile.is_tarfile(file_path):
            with tarfile.TarFile(file_path, "r") as tf:
                result = tf.getnames()
        else:
            raise ValueError("Cannot open archive file!")

        # 4: Returns the matching files.
        return {"asset_names": result}

    def run(self):
        # 1: Imports flask requirements.
        from flask import Flask, request
        from flask_cors import CORS

        # 2: Create a flask app
        flask_app = Flask(__name__)
        CORS(flask_app)

        # 3: Define the upload file endpoint
        @flask_app.post("/upload_file/")
        def upload_file():
            """Upload a file directly as form data."""
            f = request.files["file"]
            return self.upload_file(f)

        @flask_app.get("/")
        def list_files():
            return self.list_files(str(Path(self.base_dir).resolve()))

        # 5: Start the flask app while providing the `host` and `port`.
        flask_app.run(host=self.host, port=self.port, load_dotenv=False)

    def alive(self):
        """Hack: Returns whether the server is alive."""
        return self.url != ""


import requests

from lightning import LightningWork


class TestFileServer(LightningWork):
    def __init__(self, drive: Drive):
        super().__init__(cache_calls=True)
        self.drive = drive

    def run(self, file_server_url: str, first=True):
        if first:
            with open("test.txt", "w") as f:
                f.write("Some text.")

            response = requests.post(
                file_server_url + "/upload_file/",
                files={'file': open("test.txt", 'rb')}
            )
            assert response.status_code == 200
        else:
            response = requests.get(file_server_url)
            assert response.status_code == 200
            assert response.json() == {"asset_names": ["test.txt"]}


from lightning import LightningApp, LightningFlow


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        # 1: Create a drive to share data between works
        self.drive = Drive("lit://file_server")
        # 2: Create  the filer server
        self.file_server = FileServer(self.drive)
        # 3: Create the file ser
        self.test_file_server = TestFileServer(self.drive)

    def run(self):
        # 1: Start the file server.
        self.file_server.run()

        # 2: Trigger the test file server work when ready.
        if self.file_server.alive():
            # 3 Execute the test file server work.
            self.test_file_server.run(self.file_server.url)
            self.test_file_server.run(self.file_server.url, first=False)

        # 4 When both execution are successful, exit the app.
        if self.test_file_server.num_successes == 2:
            self.stop()

    def configure_layout(self):
        # Expose the file_server component
        # in the UI using its `/` endpoint.
        return {"name": "File Server", "content": self.file_server}


from lightning.app.runners import MultiProcessRuntime


def test_file_server():
    app = LightningApp(Flow())
    MultiProcessRuntime(app).dispatch()


from lightning.app.testing import run_app_in_cloud


def test_file_server_in_cloud():
    # You need to provide the directory containing the app file.
    app_dir = "docs/source-app/examples/file_server"
    with run_app_in_cloud(app_dir) as (admin_page, view_page, get_logs_fn, name):
        """# 1. `admin_page` and `view_page` are playwright Page Objects.

        # Check out https://playwright.dev/python/ doc to learn more.
        # You can click the UI and trigger actions.

        # 2. By calling logs = get_logs_fn(),
        # you get all the logs currently on the admin page.

        """
