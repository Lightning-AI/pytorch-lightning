# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# custom image (from any provider)
config= L.BuildConfig(image="gcr.io/google-samples/hello-app:1.0")
component = YourComponent(cloud_build_config=config)
app = L.LightningApp(component)
