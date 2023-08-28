# app.py
# ! pip install torch
from lightning.app import LightningWork, LightningApp, CloudCompute
import torch

class PyTorchComponent(LightningWork):
   def run(self):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      model = torch.nn.Sequential(torch.nn.Linear(1, 1),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(1, 1))
      model.to(device)
      criterion = torch.nn.MSELoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

      for step in range(10000):
         model.zero_grad()
         x = torch.tensor([0.8]).to(device)
         target = torch.tensor([1.0]).to(device)
         output = model(x)
         loss = criterion(output, target)
         print(f'step: {step}.  loss {loss}')
         loss.backward()
         optimizer.step()

compute = CloudCompute('gpu')
componet = PyTorchComponent(cloud_compute=compute)
app = LightningApp(componet)
