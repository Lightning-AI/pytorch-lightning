import os

from lightning.app import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.app.storage import Mount


class Work(LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        files = os.listdir("/content/esRedditJson/")
        for file in files:
            print(file)
        assert "esRedditJson1" in files


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_1 = Work(
            cloud_compute=CloudCompute(
                mounts=Mount(
                    source="s3://ryft-public-sample-data/esRedditJson/",
                    mount_path="/content/esRedditJson/",
                ),
            )
        )

    def run(self):
        self.work_1.run()


app = LightningApp(Flow())
