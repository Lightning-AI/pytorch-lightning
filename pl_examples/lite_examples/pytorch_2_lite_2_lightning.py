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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite

#############################################################################################
#                        Section 1: PyTorch to Lightning Lite                               #
#                                                                                           #
#                               What is LightningLite ?                                     #
#                                                                                           #
# `LightningLite` is a python class you can override to get access to Lightning             #
# accelerators and scale your training, but furthermore, it is intentend to be the safe     #
# route to fully transition to Lightning.                                                   #
#                                                                                           #
#                         Does LightningLite requires code changes ?                        #
#                                                                                           #
# `LightningLite` code changes are minimal and this tutorial will show you easy it is to    #
# convert using a BoringModel to `LightningLite`.                                           #
#                                                                                           #
#############################################################################################

#############################################################################################
#                               Pure PyTorch Section                                        #
#############################################################################################


# 1 / 6: Implement a BoringModel with only one layer.
class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


# 2 / 6: Implement a `configure_optimizers` taking a  and returning an optimizer


def configure_optimizers(module: nn.Module):
    return torch.optim.SGD(module.parameters(), lr=0.001)


# 3 / 6: Implement a simple dataset returning random data with the specificed shape


class RandomDataset(Dataset):
    def __init__(self, length: int, size: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


# 4 / 6: Implement the functions to create the dataloaders.


def train_dataloader():
    return DataLoader(RandomDataset(64, 32))


def val_dataloader():
    return DataLoader(RandomDataset(64, 32))


# 5 / 6: Our main PyTorch Loop to train our `BoringModel` on our random data.


def main(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int = 10):
    optimizer = configure_optimizers(model)

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []

        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()

        for batch in val_dataloader:
            val_losses.append(model(batch))

        train_epoch_loss = torch.stack(train_losses).mean()
        val_epoch_loss = torch.stack(val_losses).mean()

        print(f"{epoch}/{num_epochs}| Train Epoch Loss: {torch.mean(train_epoch_loss)}")
        print(f"{epoch}/{num_epochs}| Valid Epoch Loss: {torch.mean(val_epoch_loss)}")

    return model.state_dict()


# 6 / 6: Run the pure PyTorch Loop and train / validate the model.
seed_everything(42)
model = BoringModel()
pure_model_weights = main(model, train_dataloader(), val_dataloader())


#############################################################################################
#                                 Convert to LightningLite                                  #
#                                                                                           #
# By converting the `LightningLite`, you get the full power of Lightning accelerators       #
# while conversing your original code !                                                     #
# To get started, you would need to `from pytorch_lightning.lite import LightningLite`      #
# and override its `run` method.                                                            #
#############################################################################################


class LiteTrainer(LightningLite):
    def run(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int = 10):
        optimizer = configure_optimizers(model)

        ##################################################################
        # You would need to call `self.setup` to wrap `model`            #
        # and `optimizer`. If you have multiple models (c.f GAN),        #
        # call `setup` for each one of them and their associated         #
        # optimizers                                                     #
        model, optimizer = self.setup(model=model, optimizers=optimizer)  #
        ##################################################################

        for epoch in range(num_epochs):
            train_losses = []
            val_losses = []

            for batch in train_dataloader:
                optimizer.zero_grad()
                loss = model(batch)
                train_losses.append(loss)
                ##################################################################
                # By calling `self.backward` directly, `LightningLite` will      #
                # automate precision and distributions.                          #
                self.backward(loss)  #                                           #
                ##################################################################
                optimizer.step()

            for batch in val_dataloader:
                val_losses.append(model(batch))

            train_epoch_loss = torch.stack(train_losses).mean()
            val_epoch_loss = torch.stack(val_losses).mean()

            #######################################################################################
            # Optional: Utility to print only on rank 0 (when using distributed setting )        #
            self.print(f"{epoch}/{num_epochs}| Train Epoch Loss: {torch.mean(train_epoch_loss)}")  #
            self.print(f"{epoch}/{num_epochs}| Valid Epoch Loss: {torch.mean(val_epoch_loss)}")  #
            #######################################################################################


seed_everything(42)
lite_model = BoringModel()
lite = LiteTrainer()
lite.run(lite_model, train_dataloader(), val_dataloader())

#############################################################################################
#                           Assert the weights are the same                                 #
#############################################################################################

for pure_w, lite_w in zip(pure_model_weights.values(), lite_model.state_dict().values()):
    torch.equal(pure_w, lite_w)


#############################################################################################
#                                 Convert to Lightning                                      #
#                                                                                           #
# By converting to Lightning, non-only your research code becomes inter-operable            #
# (can easily be shared), but you get access to hundreds of extra features to make your     #
# research faster.                                                                          #
#############################################################################################

from pytorch_lightning import LightningDataModule, LightningModule, Trainer  # noqa E402


class LightningBoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))

    #############################################################################################
    #                                 LightningModule hooks                                     #
    #
    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        self.log("train_loss", x)
        return x

    def validation_step(self, batch, batch_idx):
        x = self.forward(batch)
        self.log("val_loss", x)
        return x

    def configure_optimizers(self):
        return configure_optimizers(self)

    #############################################################################################


class BoringDataModule(LightningDataModule):
    def train_dataloader(self):
        return train_dataloader()

    def val_dataloader(self):
        return val_dataloader()


seed_everything(42)
lightning_module = LightningBoringModel()
datamodule = BoringDataModule()
trainer = Trainer(max_epochs=10)
trainer.fit(lightning_module, datamodule)


#############################################################################################
#                           Assert the weights are the same                                 #
#############################################################################################

for pure_w, lite_w in zip(pure_model_weights.values(), lightning_module.state_dict().values()):
    torch.equal(pure_w, lite_w)
