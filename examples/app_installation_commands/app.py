# EXAMPLE COMPONENT: RUN A SCRIPT
# app.py
# !echo "I am installing a dependency not declared in a requirements file"
# !pip install lmdb
import lmdb

import lightning as L


class YourComponent(L.LightningWork):
    def run(self):
        print(lmdb.__version__)
        print("lmdb successfully installed")


# run on a cloud machine
compute = L.CloudCompute("cpu")
worker = YourComponent(cloud_compute=compute)
app = L.LightningApp(worker)
