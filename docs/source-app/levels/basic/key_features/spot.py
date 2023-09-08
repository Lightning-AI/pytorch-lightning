# app.py
from lightning.app import LightningWork, LightningApp, CloudCompute


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')

# spot machines can be turned off without notice, use for non-critical, resumable work
# request a spot machine, after 60 seconds of waiting switch to full-price
compute = CloudCompute('gpu', wait_timeout=60, spot=True)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)
