# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# stop the machine when idle for 10 seconds
compute = L.CloudCompute('gpu', idle_timeout=10)
component = YourComponent(cloud_compute=compute)
app = L.LightningApp(component)
