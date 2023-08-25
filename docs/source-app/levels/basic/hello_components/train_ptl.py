# A hello world component
# app.py
from lightning.app import LightningWork, LightningApp, CloudCompute


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')



# run on a cloud machine
compute = CloudCompute("cpu")
worker = YourComponent(cloud_compute=compute)
app = LightningApp(worker)
