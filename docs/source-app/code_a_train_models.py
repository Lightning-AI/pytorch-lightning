# app.py
import lightning as L

class LitWorker(L.LightningWork):
   def run(self):
      model = pretrained_model(...)
      Trainer(gpus=4).fit(model, ...)


# run on 1 cloud GPU
compute = L.CloudCompute("gpu-fast-multi")
app = L.LightningApp(LitWorker(cloud_compute=compute))
