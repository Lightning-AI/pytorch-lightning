# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# run on a cloud machine ("cpu", "gpu", ...)
component = YourComponent(cloud_compute=L.CloudCompute("cpu"))
app = L.LightningApp(component)
