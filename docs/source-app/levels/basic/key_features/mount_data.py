from lightning.app import LightningWork, LightningApp, CloudCompute
import os


class YourComponent(LightningWork):
   def run(self):
      os.listdir('/foo')

# mount the files on the s3 bucket under this path
mount = Mount(source="s3://lightning-example-public/", mount_path="/foo")
compute = CloudCompute(mounts=mount)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)
