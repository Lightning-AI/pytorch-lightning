# app.py
from lightning.app import LightningWork, LightningApp, CloudCompute


class YourComponent(LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')


# custom accelerators
compute = CloudCompute('gpu')
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)

# OTHER ACCELERATORS:
# compute = CloudCompute('default')          # 1 CPU
# compute = CloudCompute('cpu-medium')       # 8 CPUs
# compute = CloudCompute('gpu')              # 1 T4 GPU
# compute = CloudCompute('gpu-fast-multi')   # 4 V100 GPU
# compute = CloudCompute('p4d.24xlarge')     # AWS instance name (8 A100 GPU)
# compute = ...
