import os

import lightning as L
from lightning_app import CloudCompute
from lightning_app.storage import Mount


class Work(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        for file in os.listdir("/content/esRedditJson/1/"):
            print(file)


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_1 = Work(
            cloud_compute=CloudCompute(
                mount=Mount(
                    source="s3://ryft-public-sample-data/esRedditJson/",
                    root_dir="/content/esRedditJson/1/",
                ),
            )
        )

    def run(self):
        # Pass the drive to both works.
        self.work_1.run()
        self._exit("Application End!")


app = L.LightningApp(Flow(), debug=True)
