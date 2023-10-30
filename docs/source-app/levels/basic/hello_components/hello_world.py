# app.py
from lightning.app import LightningWork, LightningApp


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')



component = YourComponent()
app = LightningApp(component)
