# EXAMPLE COMPONENT: HELLO WORLD
# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')



# run on a cloud machine ("cpu", "gpu", ...)
compute = L.CloudCompute("cpu")
component = YourComponent(cloud_compute=compute)
app = L.LightningApp(component)
