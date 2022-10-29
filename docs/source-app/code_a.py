# app.py
import lightning as L

class LitWorker(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')



# run on a cloud machine
compute = L.CloudCompute("cpu")
worker = LitWorker(cloud_compute=compute)
app = L.LightningApp(worker)
