import os

from lightning.app import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.app.components import TracerPythonScript
from lightning.app.storage.path import Path

FILE_CONTENT = """
Hello there!
This tab is currently an IFrame of the FastAPI Server running in `DestinationFileAndServeWork`.
Also, the content of this file was created in `SourceFileWork` and then transferred to `DestinationFileAndServeWork`.
Are you already 🤯 ? Stick with us, this is only the beginning. Lightning is 🚀.
"""


class SourceFileWork(LightningWork):
    def __init__(self, cloud_compute: CloudCompute = CloudCompute(), **kwargs):
        super().__init__(parallel=True, **kwargs, cloud_compute=cloud_compute)
        self.boring_path = None

    def run(self):
        # This should be used as a REFERENCE to the file.
        self.boring_path = "lit://boring_file.txt"
        with open(self.boring_path, "w", encoding="utf-8") as f:
            f.write(FILE_CONTENT)


class DestinationFileAndServeWork(TracerPythonScript):
    def run(self, path: Path):
        assert path.exists()
        self.script_args += [f"--filepath={path}", f"--host={self.host}", f"--port={self.port}"]
        super().run()


class BoringApp(LightningFlow):
    def __init__(self):
        super().__init__()
        self.source_work = SourceFileWork()
        self.dest_work = DestinationFileAndServeWork(
            script_path=os.path.join(os.path.dirname(__file__), "scripts/serve.py"),
            port=1111,
            parallel=False,  # runs until killed.
            cloud_compute=CloudCompute(),
            raise_exception=True,
        )

    @property
    def ready(self) -> bool:
        return self.dest_work.is_running

    def run(self):
        self.source_work.run()
        if self.source_work.has_succeeded:
            # the flow passes the file from one work to another.
            self.dest_work.run(self.source_work.boring_path)
            self.stop("Boring App End")

    def configure_layout(self):
        return {"name": "Boring Tab", "content": self.dest_work.url + "/file"}


app = LightningApp(BoringApp())
