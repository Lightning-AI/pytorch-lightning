# Copyright The Lightning AI team.
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
import collections
import copy
from typing import Any

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from torch import Tensor


class TrainerStateChecker(Callback):
    def __init__(
        self,
        optimizer_dict: dict,
        model_dict: dict,
        capture: bool,
        target_device: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.optimizer_dict = optimizer_dict
        self.model_dict = model_dict
        self.capture = capture
        self.target_device = target_device

    def on_train_start(self, trainer, pl_module):
        if not self.capture:
            # Check model and optimizer device locations
            assert trainer.model.device == self.model_dict[self.target_device].device
            assert_opt_state_in_expected_location(trainer.optimizers[0], self.optimizer_dict[self.target_device])

    def on_train_end(self, trainer, pl_module):
        if self.capture:
            # Capture the optimizer state before it is transferred back to the cpu
            self.optimizer_dict[self.target_device] = copy.deepcopy(trainer.optimizers[0])
            self.model_dict[self.target_device] = copy.deepcopy(trainer.model)


def assert_opt_state_in_expected_location(opt, expected_opt):
    opt_dict = opt.state_dict()
    expected_opt_dict = expected_opt.state_dict()
    for key, param in opt_dict["state"].items():
        if isinstance(param, Tensor) and param.data.device.type != expected_opt_dict["state"][key].device.type:
            pytest.fail(f"Optimizer device mismatch for state[{key}]")
        elif isinstance(param, collections.abc.Mapping):
            for subkey, subparam in param.items():
                if (
                    isinstance(subparam, Tensor)
                    and subparam.data.device.type != expected_opt_dict["state"][key][subkey].device.type
                ):
                    pytest.fail(f"Optimizer device mismatch for state[{key}][{subkey}]")


def test_change_device(tmpdir):
    """This test validates that a generated ModelCheckpoint can be moved to a different device."""

    class ExtendedBoringModel(BoringModel):
        def __init__(
            self,
            target_device: str,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.target_device = target_device

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.layer.parameters(), lr=0.01)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    # Train on different devices to create profile of where the state Tensors are located for each device
    devices = ["cpu", "gpu"]
    optimizer_dict = {}
    model_dict = {}
    checkpoint_path = {}
    for device in devices:
        device_path = tmpdir.mkdir(device)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=device_path, filename="{epoch:02d}", save_top_k=-1
        )

        tsc = TrainerStateChecker(
            optimizer_dict=optimizer_dict, model_dict=model_dict, capture=True, target_device=device
        )
        trainer = Trainer(
            accelerator=device,
            devices=1,
            default_root_dir=device_path,
            max_epochs=1,
            limit_train_batches=12,
            limit_val_batches=6,
            limit_test_batches=12,
            callbacks=[checkpoint_callback, tsc],
            logger=False,
        )
        model = ExtendedBoringModel(device)
        trainer.fit(model)
        checkpoint_path[device] = checkpoint_callback.best_model_path

    # Cross load from checkpoint
    # That is, load CPU checkpoint, but target continuation on GPU, and vice versa
    # Expected state is checked via TrainerStateChecker using the above trainers created on GPU and CPU devices
    trainer_resume_dict = {}
    for device_idx, device in enumerate(devices):
        cross_device = devices[(device_idx + 1) % len(devices)]
        tsc = TrainerStateChecker(
            optimizer_dict=optimizer_dict, model_dict=model_dict, capture=False, target_device=cross_device
        )
        trainer = pl.Trainer(
            accelerator=cross_device,
            devices=1,
            default_root_dir=tmpdir,
            max_epochs=3,
            limit_train_batches=12,
            limit_val_batches=12,
            limit_test_batches=12,
            enable_progress_bar=False,
            callbacks=tsc,
        )
        model = ExtendedBoringModel(cross_device)
        trainer.fit(model, ckpt_path=checkpoint_path[device])  # Load checkpoint from original device
        trainer.test()
        trainer_resume_dict[cross_device] = trainer
