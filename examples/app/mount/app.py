import os

import lightning as L
from lightning_app import CloudCompute
from lightning_app.storage import Mount


class Work(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        files = os.listdir("/content/esRedditJson/")
        for file in files:
            print(file)
        assert "esRedditJson1" in files


class Flow(L.LightningFlow):
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


app = L.LightningApp(Flow())
