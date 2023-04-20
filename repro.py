import itertools
import time
from typing import cast, Union, List, Any

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from lightning import Fabric, LightningModule




class MyModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        return self.model(batch)


# _LightningModuleWrapperBase

def main():
    fabric = Fabric(accelerator="cpu", strategy="ddp", devices=2)
    fabric.launch()


    model = MyModel()
    model = fabric.setup(model)  # FabricModule(DDP(MyModel))

    # model.x = 1

    for _ in range(3):
        # out = model(torch.rand(4, 10))
        out = model.training_step(torch.rand(4, 10))

        if fabric.global_rank == 0:
            print("waiting", fabric.global_rank)
            time.sleep(10000)

        # prepare_for_backward(model._forward_module, out)
        fabric.backward(out.sum())  # .backward()
        print("running", fabric.global_rank)



if __name__ == "__main__":
    main()