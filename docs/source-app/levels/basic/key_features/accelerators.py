# app.py
import lightning as L


class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# custom accelerators
compute = L.CloudCompute('gpu')
component = YourComponent(cloud_compute=compute)
app = L.LightningApp(component)

# OTHER ACCELERATORS:
# compute = L.CloudCompute('default')          # 1 CPU
# compute = L.CloudCompute('cpu-medium')       # 8 CPUs
# compute = L.CloudCompute('gpu')              # 1 T4 GPU
# compute = L.CloudCompute('gpu-fast-multi')   # 4 V100 GPU
# compute = L.CloudCompute('p4d.24xlarge')     # AWS instance name (8 A100 GPU)
# compute = ...
