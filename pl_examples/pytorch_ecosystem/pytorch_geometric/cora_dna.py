"""Graph Convolution Example using Pytorch Geometric

This example illustrates how one could train a graph convolution model with DNA Conv
on Cora Dataset using pytorch-lightning. This example will also demonstrate how this
model can be easily torch-scripted, thanks to Pytorch Geometric.
"""
# python imports
import os
import os.path as osp
import sys
from functools import partial
from collections import namedtuple
from argparse import ArgumentParser
from typing import List, Optional, NamedTuple

# thrid parties libraries
import numpy as np
from torch import nn
import torch
from torch import Tensor
from torch.optim import Adam
import torch.nn.functional as F

# Lightning imports
from pytorch_lightning import (
    Trainer,
    LightningDataModule,
    LightningModule
)
from pytorch_lightning.metrics import Accuracy

try:
    # Pytorch Geometric imports
    from torch_geometric.nn import DNAConv, MessagePassing
    from torch_geometric.data import DataLoader
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    from torch_geometric.data import NeighborSampler
    from lightning import lightning_logo, nice_print
except Exception:
    HAS_PYTORCH_GEOMETRIC = False
else:
    HAS_PYTORCH_GEOMETRIC = True


# use to make model jittable
OptTensor = Optional[Tensor]
ListTensor = List[Tensor]


class TensorBatch(NamedTuple):
    x: Tensor
    edge_index: ListTensor
    edge_attr: OptTensor
    batch: OptTensor

###################################
#       LightningDataModule       #
###################################


class CoraDataset(LightningDataModule):

    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.
    c.f https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/planetoid.py
    """

    NAME = "cora"

    def __init__(self,
                 num_workers: int = 1,
                 batch_size: int = 8,
                 drop_last: bool = True,
                 pin_memory: bool = True,
                 num_layers: int = None):
        super().__init__()

        assert num_layers is not None

        self._num_workers = num_workers
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._pin_memory = pin_memory
        self._num_layers = num_layers

        self._transform = T.NormalizeFeatures()

    @property
    def num_features(self):
        return 1433

    @property
    def num_classes(self):
        return 7

    @property
    def hyper_parameters(self):
        # used to inform the model the dataset specifications
        return {"num_features": self.num_features, "num_classes": self.num_classes}

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )
        self.dataset = Planetoid(path, self.NAME, transform=self._transform)
        self.data = self.dataset[0]

    def create_neighbor_sampler(self, batch_size=2, stage=None):
        # https://github.com/rusty1s/pytorch_geometric/tree/master/torch_geometric/data/sampler.py#L18
        return NeighborSampler(
            self.data.edge_index,
            # the nodes that should be considered for sampling.
            node_idx=getattr(self.data, f"{stage}_mask"),
            # -1 indicates all neighbors will be selected
            sizes=[self._num_layers, -1],
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            pin_memory=self._pin_memory,
        )

    def train_dataloader(self):
        return self.create_neighbor_sampler(stage="train")

    def validation_dataloader(self):
        return self.create_neighbor_sampler(stage="val")

    def test_dataloader(self):
        return self.create_neighbor_sampler(stage="test")

    def gather_data_and_convert_to_namedtuple(self, batch, batch_nb):
        """
        This function will select features using node_idx
        and create a NamedTuple Object.
        """

        usual_keys = ["x", "edge_index", "edge_attr", "batch"]
        Batch: TensorBatch = namedtuple("Batch", usual_keys)
        return (
            Batch(
                self.data.x[batch[1]],
                [e.edge_index for e in batch[2]],
                None,
                None,
            ),
            self.data.y[batch[1]],
        )

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--drop_last", default=True)
        parser.add_argument("--pin_memory", default=True)
        return parser


###############################
#       LightningModule       #
###############################


class DNAConvNet(LightningModule):

    r"""The dynamic neighborhood aggregation operator from the `"Just Jump:
    Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"
    <https://arxiv.org/abs/1904.04849>`_ paper
    c.f https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/dna_conv.py#L172
    """

    def __init__(self,
                 num_layers: int = 2,
                 hidden_channels: int = 128,
                 heads: int = 8,
                 groups: int = 16,
                 dropout: float = 0.8,
                 cached: bool = False,
                 num_features: int = None,
                 num_classes: int = None,
                 ):
        super().__init__()

        assert num_features is not None
        assert num_classes is not None

        # utils from Lightning to save __init__ arguments
        self.save_hyperparameters()
        hparams = self.hparams

        # Instantiate metrics
        self.val_acc = Accuracy(hparams["num_classes"])
        self.test_acc = Accuracy(hparams["num_classes"])

        # Define DNA graph convolution model
        self.hidden_channels = hparams["hidden_channels"]
        self.lin1 = nn.Linear(hparams["num_features"], hparams["hidden_channels"])

        # Create ModuleList to hold all convolutions
        self.convs = nn.ModuleList()

        # Iterate through the number of layers
        for _ in range(hparams["num_layers"]):

            # Create a DNA Convolution - This graph convolution relies on MultiHead Attention mechanism
            # to route information similar to Transformers.
            # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/dna_conv.py#L172
            self.convs.append(
                DNAConv(
                    hparams["hidden_channels"],
                    hparams["heads"],
                    hparams["groups"],
                    dropout=hparams["dropout"],
                    cached=False,
                )
            )
        # classification MLP
        self.lin2 = nn.Linear(hparams["hidden_channels"], hparams["num_classes"], bias=False)

    def forward(self, batch: TensorBatch):
        # batch needs to be typed for making this model jittable.
        x = batch.x
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)

        # iterate over all convolutions
        for idx, conv in enumerate(self.convs):
            # perform convolution using previously concatenated embedding
            # through edge_index
            x = F.relu(conv(x_all, batch.edge_index[idx]))
            x = x.view(-1, 1, self.hidden_channels)

            # concatenate with previously computed embedding
            x_all = torch.cat([x_all, x], dim=1)

        # extra latest layer embedding
        x = x_all[:, -1]

        x = F.dropout(x, p=0.5, training=self.training)

        # return logits per nodes
        return F.log_softmax(self.lin2(x), -1)

    def step(self, batch, batch_nb):
        typed_batch, targets = self.gather_data_and_convert_to_namedtuple(batch, batch_nb)
        logits = self(typed_batch)
        return logits, targets

    def training_step(self, batch, batch_nb):
        logits, targets = self.step(batch, batch_nb)
        train_loss = F.nll_loss(logits, targets)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_nb):
        logits, targets = self.step(batch, batch_nb)
        val_loss = F.nll_loss(logits, targets)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, targets), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_nb):
        logits, targets = self.step(batch, batch_nb)
        test_loss = F.nll_loss(logits, targets)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc(logits, targets), on_step=False, on_epoch=True, prog_bar=True)

    # Use for jittable demonstration.

    def _convert_to_jittable(self, module):
        for key, m in module._modules.items():
            if isinstance(m, MessagePassing) and m.jittable is not None:
                # Pytorch Geometric MessagePassing implements a `.jittable` function
                # which converts the current module into its jittable version.
                module._modules[key] = m.jittable()
            else:
                self._convert_to_jittable(m)
        return module

    def jittable(self):
        for key, m in self._modules.items():
            self._modules[key] = self._convert_to_jittable(m)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--hidden_channels", type=int, default=128)
        parser.add_argument("--heads", type=int, default=8)
        parser.add_argument("--groups", type=int, default=16)
        parser.add_argument("--dropout", type=float, default=0.8)
        parser.add_argument("--cached", type=int, default=0)
        parser.add_argument("--jit", default=True)
        return parser

#################################
#     Instantiate Functions     #
#################################


def instantiate_datamodule(args):
    datamodule = CoraDataset(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory,
        num_layers=args.num_layers,
    )
    return datamodule


def instantiate_model(args, datamodule):
    model = DNAConvNet(
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        groups=args.groups,
        dropout=args.dropout,
        # provide dataset specific arguments
        **datamodule.hyper_parameters,
    )
    if args.jit:
        model.jittable()

    # Attached datamodule function to model
    model.gather_data_and_convert_to_namedtuple = datamodule.gather_data_and_convert_to_namedtuple
    return model


def get_single_batch(datamodule):
    for batch in datamodule.test_dataloader():
        return datamodule.gather_data_and_convert_to_namedtuple(batch, 0)

#######################
#     Trainer Run     #
#######################


def run(args):

    nice_print("You are about to train a TorchScripted Pytorch Geometric Lightning model !")
    nice_print(lightning_logo)

    datamodule: LightningDataModule = instantiate_datamodule(args)
    model: LightningModule = instantiate_model(args, datamodule)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)
    trainer.test()

    batch = get_single_batch(datamodule)
    model.to_torchscript(file_path="model_trace.pt",
                         method='script',
                         example_inputs=batch)

    nice_print("Congratulations !")
    nice_print("You trained your first TorchScripted Pytorch Geometric Lightning model !", last=True)


if __name__ == "__main__":
    if not HAS_PYTORCH_GEOMETRIC:
        print("Skip training. Pytorch Geometric isn't installed. Please, check README.md !")

    else:
        parser = ArgumentParser(description="Pytorch Geometric Example")
        parser = Trainer.add_argparse_args(parser)
        parser = CoraDataset.add_argparse_args(parser)
        parser = DNAConvNet.add_argparse_args(parser)

        cmd_line = '--max_epochs 1'.split(' ')

        run(parser.parse_args(cmd_line))
