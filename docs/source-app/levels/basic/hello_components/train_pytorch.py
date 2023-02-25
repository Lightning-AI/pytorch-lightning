# app.py
# ! pip install torch
import lightning as L
import torch

class PyTorchComponent(L.LightningWork):
   def run(self):
      if torch.cuda.is_available():
          device = torch.device("cuda:0")
      elif torch.xpu.is_available():
          device = torch.device("xpu:0")
      else:
          device = torch.device("cpu")
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

compute = L.CloudCompute('gpu')
componet = PyTorchComponent(cloud_compute=compute)
app = L.LightningApp(componet)
