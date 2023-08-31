# app.py
from lightning.app import LightningWork, LightningApp, CloudCompute


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# use 100 GB of space on that machine (max size: 64 TB)
compute = CloudCompute('gpu', disk_size=100)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)
