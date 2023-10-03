# EXAMPLE COMPONENT: RUN A SCRIPT
# app.py
# !echo "I am installing a dependency not declared in a requirements file"
# !pip install lmdb
import lmdb
from lightning.app import CloudCompute, LightningApp, LightningFlow, LightningWork


class YourComponent(LightningWork):
    def run(self):
        print(lmdb.version())
        print("lmdb successfully installed")
        print("Accessing a module in a Work or Flow body works!")


class RootFlow(LightningFlow):
    def __init__(self, work):
        super().__init__()
        self.work = work

    def run(self):
        self.work.run()


print(f"Accessing an object in main code body works!: version = {lmdb.version()}")


# run on a cloud machine
compute = CloudCompute("cpu")
worker = YourComponent(cloud_compute=compute)
app = LightningApp(RootFlow(worker))
