# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# use 100 GB of space on that machine (max size: 64 TB)
compute = L.CloudCompute('gpu', disk_size=100)
component = YourComponent(cloud_compute=compute)
app = L.LightningApp(component)
