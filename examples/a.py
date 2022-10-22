import os
import lightning as L
from lightning.app.storage import Mount

class LitWorker(L.LightningWork):
   def run(self):
      os.listdir('/foo')
      file = os.file('/foo/a.jpg')

mount = Mount(source="s3://lightning-example-public/", mount_path="/foo")
compute = L.CloudCompute(mounts=mount)

app = L.LightningApp(LitWorker(cloud_compute=compute))
