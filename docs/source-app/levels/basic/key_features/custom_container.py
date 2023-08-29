# app.py
from lightning.app import LightningWork, LightningApp


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# custom image (from any provider)
config= BuildConfig(image="gcr.io/google-samples/hello-app:1.0")
component = YourComponent(cloud_build_config=config)
app = LightningApp(component)
