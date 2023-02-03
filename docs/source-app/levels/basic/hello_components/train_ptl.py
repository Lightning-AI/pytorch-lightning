# A hello world component
# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')



# run on a cloud machine
compute = L.CloudCompute("cpu")
worker = YourComponent(cloud_compute=compute)
app = L.LightningApp(worker)
