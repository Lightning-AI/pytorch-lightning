# app.py
import lightning as L

class LitWorker(L.LightningWork):
   def run(self):
      model = pretrained_model(...)
      app = FastAPI()
      app.post("/predict/")(your_predict_function(models))

# run on 1 cloud GPU
compute = L.CloudCompute("gpu-fast-multi")
app = L.LightningApp(LitWorker(cloud_compute=compute))
