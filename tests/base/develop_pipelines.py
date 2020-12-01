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
from functools import singledispatch

import torch
from pytorch_lightning import Trainer
from tests.base import BoringModel
from tests.base.develop_utils import get_default_logger, load_model_from_checkpoint, reset_seed


def run_model_test_without_loggers(trainer_options, model, min_acc: float = 0.50):
    reset_seed()

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    pretrained_model = load_model_from_checkpoint(
        trainer.logger,
        trainer.checkpoint_callback.best_model_path,
        type(model)
    )

    # test new model accuracy
    test_loaders = model.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    for dataloader in test_loaders:
        run_prediction(pretrained_model, dataloader, min_acc=min_acc)

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()


def run_model_test(trainer_options, model, on_gpu: bool = True, version=None,
                   with_hpc: bool = True, min_acc: float = 0.25):

    reset_seed()
    save_dir = trainer_options['default_root_dir']

    # logger file to get meta
    logger = get_default_logger(save_dir, version=version)
    trainer_options.update(logger=logger)

    # TODO: DEPRECATED option
    if "checkpoint_callback" not in trainer_options:
        trainer_options.update(checkpoint_callback=True)

    trainer = Trainer(**trainer_options)
    initial_values = torch.tensor([torch.sum(torch.abs(x)) for x in model.parameters()])
    result = trainer.fit(model)
    post_train_values = torch.tensor([torch.sum(torch.abs(x)) for x in model.parameters()])

    assert result == 1, 'trainer failed'
    # Check that the model is actually changed post-training
    change_ratio = torch.norm(initial_values - post_train_values)
    assert change_ratio > 0.1, f"the model is changed of {change_ratio}"

    # test model loading
    pretrained_model = load_model_from_checkpoint(logger, trainer.checkpoint_callback.best_model_path, type(model))

    # test new model accuracy
    test_loaders = model.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    for dataloader in test_loaders:
        run_prediction(pretrained_model, dataloader, min_acc=min_acc)

    if with_hpc:
        if trainer.use_ddp or trainer.use_ddp2:
            # on hpc this would work fine... but need to hack it for the purpose of the test
            trainer.model = pretrained_model
            trainer.optimizers, trainer.lr_schedulers, trainer.optimizer_frequencies = trainer.init_optimizers(
                pretrained_model
            )

        # test HPC saving
        trainer.checkpoint_connector.hpc_save(save_dir, logger)
        # test HPC loading
        checkpoint_path = trainer.checkpoint_connector.get_max_ckpt_path_from_folder(save_dir)
        trainer.checkpoint_connector.hpc_load(checkpoint_path, on_gpu=on_gpu)


@singledispatch
def run_prediction(trained_model, dataloader, dp=False, min_acc=0.50):
    # run prediction on 1 batch
    batch = next(iter(dataloader))
    x, y = batch
    x = x.view(x.size(0), -1)

    if dp:
        with torch.no_grad():
            output = trained_model(batch, 0)
        acc = output['val_acc']
        acc = torch.mean(acc).item()

    else:
        with torch.no_grad():
            y_hat = trained_model(x)
        y_hat = y_hat.cpu()

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)

        y = y.cpu()
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        acc = torch.tensor(acc)
        acc = acc.item()

    assert acc >= min_acc, f"This model is expected to get > {min_acc} in test set (it got {acc})"


@run_prediction.register(BoringModel)
def _(trained_model, dataloader, dp=False, min_acc=0.25):
    # run prediction on 1 batch
    batch = next(iter(dataloader))
    with torch.no_grad():
        output = trained_model(batch)
    acc = trained_model.loss(batch, output)

    assert acc >= min_acc, f"This model is expected to get, {min_acc} in test set but got {acc}"
