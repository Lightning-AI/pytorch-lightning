# EXAMPLE COMPONENT: RUN A SCRIPT
# app.py
# !echo "I am installing a dependency not declared in a requirements file"
# !pip install lmdb
import lmdb

import lightning as L


class YourComponent(L.LightningWork):
    def run(self):
        print(lmdb.version())
        print("lmdb successfully installed")
        print("Accessing a module in a Work or Flow body works!")


class RootFlow(L.LightningFlow):
    def __init__(self, work):
        super().__init__()
        self.work = work

    def run(self):
        self.work.run()


print(f"Accessing an object in main code body works!: version = {lmdb.version()}")


# run on a cloud machine
compute = L.CloudCompute("cpu")
worker = YourComponent(cloud_compute=compute)
app = L.LightningApp(RootFlow(worker))
