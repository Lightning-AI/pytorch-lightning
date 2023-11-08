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
from functools import partial

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_11 as _TM_GE_0_11
from torchmetrics.functional import accuracy

from tests_pytorch.helpers.utils import get_default_logger, load_model_from_checkpoint


def run_model_test_without_loggers(
    trainer_options: dict, model: LightningModule, data: LightningDataModule = None, min_acc: float = 0.50
):
    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, datamodule=data)

    # correct result and ok accuracy
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    model2 = load_model_from_checkpoint(trainer.checkpoint_callback.best_model_path, type(model))

    # test new model accuracy
    test_loaders = model2.test_dataloader() if not data else data.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    if not isinstance(model2, BoringModel):
        for dataloader in test_loaders:
            run_model_prediction(model2, dataloader, min_acc=min_acc)


def run_model_test(
    trainer_options,
    model: LightningModule,
    data: LightningDataModule = None,
    version=None,
    with_hpc: bool = True,
    min_acc: float = 0.25,
    min_change_ratio: float = 0.03,
):
    save_dir = trainer_options["default_root_dir"]

    # logger file to get meta
    logger = get_default_logger(save_dir, version=version)
    trainer_options.update(logger=logger)
    trainer = Trainer(**trainer_options)
    with torch.no_grad():
        initial_values = torch.cat([x.view(-1) for x in model.parameters()])
    trainer.fit(model, datamodule=data)
    with torch.no_grad():
        post_train_values = torch.cat([x.view(-1) for x in model.parameters()])

    # Check that the model has changed post-training
    change_ratio = torch.norm(initial_values - post_train_values) / torch.norm(initial_values)
    assert change_ratio >= min_change_ratio, (
        f"The change in the model's parameter norm is {change_ratio:.1f}"
        f" relative to the initial norm, but expected a change by >={min_change_ratio}"
    )

    if trainer.world_size != trainer.num_devices:
        # we're in multinode. unless the filesystem is shared, only the main node will have access to the checkpoint
        # since we cannot know this, the code below needs to be skipped
        return

    # test model loading
    _ = load_model_from_checkpoint(trainer.checkpoint_callback.best_model_path, type(model))

    # test new model accuracy
    test_loaders = model.test_dataloader() if not data else data.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    if not isinstance(model, BoringModel):
        for dataloader in test_loaders:
            run_model_prediction(model, dataloader, min_acc=min_acc)

    if with_hpc:
        # test HPC saving
        # save logger to make sure we get all the metrics
        if logger:
            logger.finalize("finished")
        hpc_save_path = trainer._checkpoint_connector.hpc_save_path(save_dir)
        trainer.save_checkpoint(hpc_save_path)
        # test HPC loading
        checkpoint_path = trainer._checkpoint_connector._CheckpointConnector__get_max_ckpt_path_from_folder(save_dir)
        trainer._checkpoint_connector.restore(checkpoint_path)


@torch.no_grad()
def run_model_prediction(trained_model, dataloader, min_acc=0.50):
    orig_device = trained_model.device
    # run prediction on 1 batch
    trained_model.cpu()
    trained_model.eval()

    batch = next(iter(dataloader))
    x, y = batch
    x = x.flatten(1)

    y_hat = trained_model(x)
    metric = partial(accuracy, task="multiclass") if _TM_GE_0_11 else accuracy
    acc = metric(y_hat.cpu(), y.cpu(), top_k=2, num_classes=y_hat.size(-1)).item()

    assert acc >= min_acc, f"This model is expected to get > {min_acc} in test set (it got {acc})"
    trained_model.to(orig_device)
