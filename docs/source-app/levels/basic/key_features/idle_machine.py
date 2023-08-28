# app.py
from lightning.app import LightningWork, LightningApp, CloudCompute


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# stop the machine when idle for 10 seconds
compute = CloudCompute('gpu', idle_timeout=10)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)
