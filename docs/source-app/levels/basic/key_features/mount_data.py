import lightning as L
import os


class YourComponent(L.LightningWork):
   def run(self):
      os.listdir('/foo')

# mount the files on the s3 bucket under this path
mount = L.Mount(source="s3://lightning-example-public/", mount_path="/foo")
compute = L.CloudCompute(mounts=mount)
component = YourComponent(cloud_compute=compute)
app = L.LightningApp(component)
