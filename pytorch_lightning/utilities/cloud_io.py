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

import io
from distutils.version import LooseVersion
from pathlib import Path
from typing import IO, Union

import fsspec
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import (
    _APEX_AVAILABLE,
    _OMEGACONF_AVAILABLE,
    AMPType,
    DeviceType,
)

if _APEX_AVAILABLE:
    from apex import amp

if _OMEGACONF_AVAILABLE:
    from omegaconf import Container


def load(path_or_url: Union[str, IO, Path], map_location=None):
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similiar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def get_filesystem(path: Union[str, Path]):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0])
    else:
        # use local filesystem
        return fsspec.filesystem("file")


def atomic_save(checkpoint, filepath: str):
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """

    bytesbuffer = io.BytesIO()
    # Can't use the new zipfile serialization for 1.6.0 because there's a bug in
    # torch.hub.load_state_dict_from_url() that prevents it from loading the new files.
    # More details can be found here: https://github.com/pytorch/pytorch/issues/42239
    if LooseVersion(torch.__version__).version[:3] == [1, 6, 0]:
        torch.save(checkpoint, bytesbuffer, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, bytesbuffer)
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())


def dump_checkpoint(trainer: 'pl.Trainer', weights_only: bool = False) -> dict:
    """Creating a model checkpoint dictionary object from various component states.

    Args:
        weights_only: saving model weights only

    Return:
        structured dictionary: {
            'epoch':                     training epoch
            'global_step':               training global step
            'pytorch-lightning_version': PyTorch Lightning's version
            'callbacks':                 "callback specific state"[] # if not weights_only
            'optimizer_states':          "PT optim's state_dict"[]   # if not weights_only
            'lr_schedulers':             "PT sched's state_dict"[]   # if not weights_only
            'native_amp_scaling_state':  PT amp's state_dict         # if not weights_only and use native amp
            'amp_scaling_state':         Apex's state_dict           # if not weights_only and use apex amp
            'state_dict':                Model's state_dict (e.g. network weights)
            CHECKPOINT_HYPER_PARAMS_NAME:
            CHECKPOINT_HYPER_PARAMS_KEY:
            CHECKPOINT_HYPER_PARAMS_TYPE:
            something_cool_i_want_to_save: anything you define through model.on_save_checkpoint
            LightningDataModule.__class__.__name__: pl DataModule's state
        }
    """
    from pytorch_lightning import LightningModule

    # dump epoch/global_step/pytorch-lightning_version
    current_epoch = trainer.current_epoch
    global_step = trainer.global_step
    has_reached_max_steps = trainer.max_steps and trainer.max_steps <= global_step

    global_step += 1
    if not has_reached_max_steps:
        current_epoch += 1

    model = trainer.lightning_module

    checkpoint = {
        'epoch': current_epoch,
        'global_step': global_step,
        'pytorch-lightning_version': pl.__version__,
        'state_dict': model.state_dict(),
    }

    if not weights_only:
        # dump callbacks
        checkpoint['callbacks'] = trainer.on_save_checkpoint(checkpoint)

        optimizer_states = []
        for i, optimizer in enumerate(trainer.optimizers):
            # Rely on accelerator to dump optimizer state
            optimizer_state = trainer.accelerator.optimizer_state(optimizer)
            optimizer_states.append(optimizer_state)

        checkpoint['optimizer_states'] = optimizer_states

        # dump lr schedulers
        lr_schedulers = []
        for scheduler in trainer.lr_schedulers:
            lr_schedulers.append(scheduler['scheduler'].state_dict())
        checkpoint['lr_schedulers'] = lr_schedulers

        # dump amp scaling
        if (
            trainer.amp_backend == AMPType.NATIVE and trainer._device_type != DeviceType.TPU
            and trainer.scaler is not None
        ):
            checkpoint['native_amp_scaling_state'] = trainer.scaler.state_dict()
        elif trainer.amp_backend == AMPType.APEX:
            checkpoint['amp_scaling_state'] = amp.state_dict()

    # dump hyper-parameters
    if model.hparams:
        if hasattr(model, '_hparams_name'):
            checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_NAME] = model._hparams_name
        # dump arguments
        if _OMEGACONF_AVAILABLE and isinstance(model.hparams, Container):
            checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY] = model.hparams
            checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_TYPE] = type(model.hparams)
        else:
            checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY] = dict(model.hparams)

    # give the model a chance to dump a few things
    model.on_save_checkpoint(checkpoint)
    if trainer.datamodule is not None:
        trainer.datamodule.on_save_checkpoint(checkpoint)

    return checkpoint
