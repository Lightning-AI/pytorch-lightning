# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from tests.helpers.boring_model import BoringModel


class ManualOptModel(BoringModel):

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt_a, opt_b = self.optimizers()

        # make sure there are no grads
        if batch_idx > 0:
            assert torch.all(self.layer.weight.grad == 0)

        loss_1 = self.step(batch[0])
        self.manual_backward(loss_1)
        opt_a.step()
        opt_a.zero_grad()
        assert torch.all(self.layer.weight.grad == 0)

        loss_2 = self.step(batch[0])
        # ensure we forward the correct params to the optimizer
        # without retain_graph we can't do multiple backward passes
        self.manual_backward(loss_2, retain_graph=True)
        self.manual_backward(loss_2)
        assert self.layer.weight.grad is not None
        opt_b.step()
        opt_b.zero_grad()
        assert torch.all(self.layer.weight.grad == 0)

        return loss_2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return optimizer, optimizer_2
