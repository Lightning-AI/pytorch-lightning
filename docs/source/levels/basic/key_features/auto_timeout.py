# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# if the machine hasn't started after 60 seconds, cancel the work
compute = L.CloudCompute('gpu', wait_timeout=60)
component = YourComponent(cloud_compute=compute)
app = L.LightningApp(component)
