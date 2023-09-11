# app.py
from lightning.app import LightningWork, LightningApp, CloudCompute


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# if the machine hasn't started after 60 seconds, cancel the work
compute = CloudCompute('gpu', wait_timeout=60)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)
