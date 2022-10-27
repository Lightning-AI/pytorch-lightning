# app.py
import lightning as L

class LitWorker(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')



# run on 1 cloud GPU
compute = L.CloudCompute("gpu")
app = L.LightningApp(LitWorker(cloud_compute=compute))
