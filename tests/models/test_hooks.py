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
from functools import partial
from inspect import getmembers, isfunction
from unittest import mock
from unittest.mock import ANY, PropertyMock

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import __version__, Callback, LightningDataModule, LightningModule, Trainer
from tests.helpers import BoringDataModule, BoringModel, RandomDataset
from tests.helpers.runif import RunIf


@pytest.mark.parametrize("max_steps", [1, 2, 3])
def test_on_before_zero_grad_called(tmpdir, max_steps):
    class CurrentTestModel(BoringModel):
        on_before_zero_grad_called = 0

        def on_before_zero_grad(self, optimizer):
            self.on_before_zero_grad_called += 1

    model = CurrentTestModel()

    trainer = Trainer(default_root_dir=tmpdir, max_steps=max_steps, max_epochs=2)
    assert 0 == model.on_before_zero_grad_called
    trainer.fit(model)
    assert max_steps == model.on_before_zero_grad_called

    model.on_before_zero_grad_called = 0
    trainer.test(model)
    assert 0 == model.on_before_zero_grad_called


def test_training_epoch_end_metrics_collection(tmpdir):
    """Test that progress bar metrics also get collected at the end of an epoch."""
    num_epochs = 3

    class CurrentModel(BoringModel):
        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            self.log_dict({"step_metric": torch.tensor(-1), "shared_metric": 100}, logger=False, prog_bar=True)
            return output

        def training_epoch_end(self, outputs):
            epoch = self.current_epoch
            # both scalar tensors and Python numbers are accepted
            self.log_dict(
                {f"epoch_metric_{epoch}": torch.tensor(epoch), "shared_metric": 111}, logger=False, prog_bar=True
            )

    model = CurrentModel()
    trainer = Trainer(max_epochs=num_epochs, default_root_dir=tmpdir, overfit_batches=2)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    metrics = trainer.progress_bar_dict

    # metrics added in training step should be unchanged by epoch end method
    assert metrics["step_metric"] == -1
    # a metric shared in both methods gets overwritten by epoch_end
    assert metrics["shared_metric"] == 111
    # metrics are kept after each epoch
    for i in range(num_epochs):
        assert metrics[f"epoch_metric_{i}"] == i


def test_training_epoch_end_metrics_collection_on_override(tmpdir):
    """Test that batch end metrics are collected when training_epoch_end is overridden at the end of an epoch."""

    class OverriddenModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.len_outputs = 0

        def on_train_epoch_start(self):
            self.num_train_batches = 0

        def training_epoch_end(self, outputs):
            self.len_outputs = len(outputs)

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.num_train_batches += 1

    class NotOverriddenModel(BoringModel):
        def on_train_epoch_start(self):
            self.num_train_batches = 0

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.num_train_batches += 1

    overridden_model = OverriddenModel()
    not_overridden_model = NotOverriddenModel()
    not_overridden_model.training_epoch_end = None

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir, overfit_batches=2)

    trainer.fit(overridden_model)
    assert overridden_model.len_outputs == overridden_model.num_train_batches


@RunIf(min_gpus=1)
@mock.patch("pytorch_lightning.accelerators.accelerator.Accelerator.lightning_module", new_callable=PropertyMock)
def test_apply_batch_transfer_handler(model_getter_mock):
    expected_device = torch.device("cuda", 0)

    class CustomBatch:
        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestModel(BoringModel):
        rank = 0
        transfer_batch_to_device_hook_rank = None
        on_before_batch_transfer_hook_rank = None
        on_after_batch_transfer_hook_rank = None

        def on_before_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx is None
            self.on_before_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.samples += 1
            return batch

        def on_after_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx is None
            assert batch.samples.device == batch.targets.device == expected_device
            self.on_after_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.targets *= 2
            return batch

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            assert dataloader_idx is None
            self.transfer_batch_to_device_hook_rank = self.rank
            self.rank += 1
            batch.samples = batch.samples.to(device)
            batch.targets = batch.targets.to(device)
            return batch

    model = CurrentTestModel()
    batch = CustomBatch((torch.zeros(5, 32), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(gpus=1)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead

    model_getter_mock.return_value = model
    batch_gpu = trainer.accelerator.batch_to_device(batch, expected_device)

    assert model.on_before_batch_transfer_hook_rank == 0
    assert model.transfer_batch_to_device_hook_rank == 1
    assert model.on_after_batch_transfer_hook_rank == 2
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected_device
    assert torch.allclose(batch_gpu.samples.cpu(), torch.ones(5, 32))
    assert torch.allclose(batch_gpu.targets.cpu(), torch.ones(5, 1, dtype=torch.long) * 2)


@RunIf(min_gpus=2, special=True)
def test_transfer_batch_hook_ddp(tmpdir):
    """
    Test custom data are properly moved to the right device using ddp
    """

    class CustomBatch:
        def __init__(self, data):
            self.samples = data[0]

        def to(self, device, **kwargs):
            self.samples = self.samples.to(device, **kwargs)
            return self

    def collate_fn(batch):
        return CustomBatch(batch)

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            assert batch.samples.device == self.device
            assert isinstance(batch_idx, int)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(RandomDataset(32, 64), collate_fn=collate_fn)

    model = TestModel()
    model.validation_step = None
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=1,
        weights_summary=None,
        accelerator="ddp",
        gpus=2,
    )
    trainer.fit(model)


def get_members(cls):
    return {h for h, _ in getmembers(cls, predicate=isfunction) if not h.startswith("_")}


class HookedCallback(Callback):
    def __init__(self, called):
        def call(hook, *args, **kwargs):
            d = {"name": f"Callback.{hook}"}
            if args:
                d["args"] = args
            if kwargs:
                d["kwargs"] = kwargs
            called.append(d)

        for h in get_members(Callback):
            setattr(self, h, partial(call, h))


class HookedModel(BoringModel):
    def __init__(self, called):
        super().__init__()
        pl_module_hooks = get_members(LightningModule)
        # remove non-hooks
        pl_module_hooks.difference_update({"optimizers"})
        # remove most `nn.Module` hooks
        module_hooks = get_members(torch.nn.Module)
        module_hooks.difference_update({"forward", "zero_grad", "train"})
        pl_module_hooks.difference_update(module_hooks)

        def call(hook, fn, *args, **kwargs):
            out = fn(*args, **kwargs)
            d = {"name": hook}
            if args:
                d["args"] = args
            elif hook == "train":
                # DeepSpeed calls `train(mode)` but we do not. Standardize
                # https://github.com/microsoft/DeepSpeed/pull/571
                d["args"] = (True,)
            if kwargs:
                d["kwargs"] = kwargs
            called.append(d)
            return out

        for h in pl_module_hooks:
            attr = getattr(self, h)
            setattr(self, h, partial(call, h, attr))

    def validation_epoch_end(self, *args, **kwargs):
        # `BoringModel` does not have a return for `validation_step_end` so this would fail
        pass

    def test_epoch_end(self, *args, **kwargs):
        # `BoringModel` does not have a return for `test_step_end` so this would fail
        pass

    def _train_batch(self, *args, **kwargs):
        if self.automatic_optimization:
            return self._auto_train_batch(*args, **kwargs)
        return self._manual_train_batch(*args, **kwargs)

    @staticmethod
    def _auto_train_batch(trainer, model, batches, device=torch.device("cpu"), current_epoch=0, **kwargs):
        using_native_amp = kwargs.get("amp_backend") == "native"
        using_deepspeed = kwargs.get("plugins") == "deepspeed"
        using_plugin = kwargs.get("amp_backend") or kwargs.get("plugins")
        out = []
        on_before_optimizer_step = [
            dict(name="Callback.on_before_optimizer_step", args=(trainer, model, ANY, 0)),
            dict(name="on_before_optimizer_step", args=(ANY, 0)),
        ]
        for i in range(batches):
            out.extend(
                [
                    dict(name="on_before_batch_transfer", args=(ANY, 0)),
                    dict(name="transfer_batch_to_device", args=(ANY, device, 0)),
                    dict(name="on_after_batch_transfer", args=(ANY, 0)),
                    # TODO: `on_batch_{start,end}`
                    dict(name="Callback.on_batch_start", args=(trainer, model)),
                    dict(name="Callback.on_train_batch_start", args=(trainer, model, ANY, i, 0)),
                    dict(name="on_train_batch_start", args=(ANY, i, 0)),
                    # without a precision plugin, we execute the closure inside the `optimizer.step`
                    *([] if using_plugin else on_before_optimizer_step),
                    dict(name="forward", args=(ANY,)),
                    dict(name="training_step", args=(ANY, i)),
                    dict(name="training_step_end", args=(dict(loss=ANY),)),
                    dict(name="Callback.on_before_zero_grad", args=(trainer, model, ANY)),
                    dict(name="on_before_zero_grad", args=(ANY,)),
                    dict(name="optimizer_zero_grad", args=(current_epoch, i, ANY, 0)),
                    dict(name="Callback.on_before_backward", args=(trainer, model, ANY)),
                    dict(name="on_before_backward", args=(ANY,)),
                    # DeepSpeed handles backward internally
                    *([dict(name="backward", args=(ANY, ANY, 0))] if not using_deepspeed else []),
                    dict(name="Callback.on_after_backward", args=(trainer, model)),
                    dict(name="on_after_backward"),
                    *(on_before_optimizer_step if using_plugin else []),
                    dict(
                        name="optimizer_step",
                        args=(current_epoch, i, ANY, 0, ANY),
                        kwargs=dict(on_tpu=False, using_lbfgs=False, using_native_amp=using_native_amp),
                    ),
                    dict(name="Callback.on_train_batch_end", args=(trainer, model, dict(loss=ANY), ANY, i, 0)),
                    dict(name="on_train_batch_end", args=(dict(loss=ANY), ANY, i, 0)),
                    dict(name="Callback.on_batch_end", args=(trainer, model)),
                ]
            )
        return out

    @staticmethod
    def _manual_train_batch(trainer, model, batches, device=torch.device("cpu"), **kwargs):
        using_deepspeed = kwargs.get("plugins") == "deepspeed"
        using_plugin = kwargs.get("amp_backend") or kwargs.get("plugins")
        out = []
        for i in range(batches):
            out.extend(
                [
                    dict(name="on_before_batch_transfer", args=(ANY, 0)),
                    dict(name="transfer_batch_to_device", args=(ANY, device, 0)),
                    dict(name="on_after_batch_transfer", args=(ANY, 0)),
                    # TODO: `on_batch_{start,end}`
                    dict(name="Callback.on_batch_start", args=(trainer, model)),
                    dict(name="Callback.on_train_batch_start", args=(trainer, model, ANY, i, 0)),
                    dict(name="on_train_batch_start", args=(ANY, i, 0)),
                    dict(name="forward", args=(ANY,)),
                    dict(name="Callback.on_before_backward", args=(trainer, model, ANY)),
                    dict(name="on_before_backward", args=(ANY,)),
                    # DeepSpeed handles backward internally
                    *([dict(name="backward", args=(ANY, None, None))] if not using_deepspeed else []),
                    dict(name="Callback.on_after_backward", args=(trainer, model)),
                    dict(name="on_after_backward"),
                    # `manual_backward` calls the previous 3
                    dict(name="manual_backward", args=(ANY,)),
                    *([dict(name="closure")] if using_plugin else []),
                    dict(name="Callback.on_before_optimizer_step", args=(trainer, model, ANY, 0)),
                    dict(name="on_before_optimizer_step", args=(ANY, 0)),
                    # without a precision plugin, we execute the closure inside the `optimizer.step`
                    *([] if using_plugin else [dict(name="closure")]),
                    dict(name="training_step", args=(ANY, i)),
                    dict(name="training_step_end", args=(dict(loss=ANY),)),
                    dict(name="Callback.on_train_batch_end", args=(trainer, model, dict(loss=ANY), ANY, i, 0)),
                    dict(name="on_train_batch_end", args=(dict(loss=ANY), ANY, i, 0)),
                    dict(name="Callback.on_batch_end", args=(trainer, model)),
                ]
            )
        return out

    @staticmethod
    def _eval_epoch(fn, trainer, model, batches, key, device=torch.device("cpu")):
        outputs = {key: ANY}
        return [
            dict(name="Callback.on_epoch_start", args=(trainer, model)),
            dict(name="on_epoch_start"),
            dict(name=f"Callback.on_{fn}_epoch_start", args=(trainer, model)),
            dict(name=f"on_{fn}_epoch_start"),
            *HookedModel._eval_batch(fn, trainer, model, batches, key, device=device),
            dict(name=f"{fn}_epoch_end", args=([outputs] * batches,)),
            dict(name=f"Callback.on_{fn}_epoch_end", args=(trainer, model)),
            dict(name=f"on_{fn}_epoch_end"),
            dict(name="Callback.on_epoch_end", args=(trainer, model)),
            dict(name="on_epoch_end"),
        ]

    @staticmethod
    def _eval_batch(fn, trainer, model, batches, key, device=torch.device("cpu")):
        out = []
        outputs = {key: ANY}
        for i in range(batches):
            out.extend(
                [
                    dict(name="on_before_batch_transfer", args=(ANY, 0)),
                    dict(name="transfer_batch_to_device", args=(ANY, device, 0)),
                    dict(name="on_after_batch_transfer", args=(ANY, 0)),
                    # TODO: `{,Callback}.on_batch_{start,end}`
                    dict(name=f"Callback.on_{fn}_batch_start", args=(trainer, model, ANY, i, 0)),
                    dict(name=f"on_{fn}_batch_start", args=(ANY, i, 0)),
                    dict(name="forward", args=(ANY,)),
                    dict(name=f"{fn}_step", args=(ANY, i)),
                    dict(name=f"{fn}_step_end", args=(outputs,)),
                    dict(name=f"Callback.on_{fn}_batch_end", args=(trainer, model, outputs, ANY, i, 0)),
                    dict(name=f"on_{fn}_batch_end", args=(outputs, ANY, i, 0)),
                ]
            )
        return out

    @staticmethod
    def _predict_batch(trainer, model, batches):
        out = []
        for i in range(batches):
            out.extend(
                [
                    dict(name="on_before_batch_transfer", args=(ANY, 0)),
                    dict(name="transfer_batch_to_device", args=(ANY, torch.device("cpu"), 0)),
                    dict(name="on_after_batch_transfer", args=(ANY, 0)),
                    # TODO: `{,Callback}.on_batch_{start,end}`
                    dict(name="Callback.on_predict_batch_start", args=(trainer, model, ANY, i, 0)),
                    dict(name="on_predict_batch_start", args=(ANY, i, 0)),
                    dict(name="forward", args=(ANY,)),
                    dict(name="predict_step", args=(ANY, i)),
                    # TODO: `predict_step_end`
                    dict(name="Callback.on_predict_batch_end", args=(trainer, model, ANY, ANY, i, 0)),
                    dict(name="on_predict_batch_end", args=(ANY, ANY, i, 0)),
                ]
            )
        return out


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        # these precision plugins modify the optimization flow, so testing them explicitly
        pytest.param(dict(gpus=1, precision=16, plugins="deepspeed"), marks=RunIf(deepspeed=True, min_gpus=1)),
        pytest.param(dict(gpus=1, precision=16, amp_backend="native"), marks=RunIf(amp_native=True, min_gpus=1)),
        pytest.param(dict(gpus=1, precision=16, amp_backend="apex"), marks=RunIf(amp_apex=True, min_gpus=1)),
    ],
)
@pytest.mark.parametrize("automatic_optimization", (True, False))
def test_trainer_model_hook_system_fit(tmpdir, kwargs, automatic_optimization):
    called = []

    class TestModel(HookedModel):
        def __init__(self, *args):
            super().__init__(*args)
            self.automatic_optimization = automatic_optimization

        def training_step(self, batch, batch_idx):
            if self.automatic_optimization:
                return super().training_step(batch, batch_idx)
            loss = self.step(batch[0])
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step(lambda: called.append({"name": "closure"}))
            return {"loss": loss}

    model = TestModel(called)
    callback = HookedCallback(called)
    train_batches = 2
    val_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        callbacks=[callback],
        **kwargs,
    )

    assert called == [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
    ]

    trainer.fit(model)

    saved_ckpt = {
        "callbacks": ANY,
        "epoch": 1,
        "global_step": train_batches,
        "lr_schedulers": ANY,
        "optimizer_states": ANY,
        "pytorch-lightning_version": __version__,
        "state_dict": ANY,
    }
    if kwargs.get("amp_backend") == "native":
        saved_ckpt["native_amp_scaling_state"] = ANY
    elif kwargs.get("amp_backend") == "apex":
        saved_ckpt["amp_scaling_state"] = ANY
    device = torch.device("cuda:0" if "gpus" in kwargs else "cpu")

    expected = [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
        dict(name="prepare_data"),
        dict(name="configure_callbacks"),
        dict(name="Callback.on_before_accelerator_backend_setup", args=(trainer, model)),
        # DeepSpeed needs the batch size to figure out throughput logging
        *([dict(name="train_dataloader")] if kwargs.get("plugins") == "deepspeed" else []),
        dict(name="Callback.setup", args=(trainer, model), kwargs=dict(stage="fit")),
        dict(name="setup", kwargs=dict(stage="fit")),
        dict(name="configure_sharded_model"),
        dict(name="Callback.on_configure_sharded_model", args=(trainer, model)),
        # DeepSpeed skips initializing optimizers here as they are handled via config
        *([dict(name="configure_optimizers")] if kwargs.get("plugins") != "deepspeed" else []),
        dict(name="Callback.on_fit_start", args=(trainer, model)),
        dict(name="on_fit_start"),
        # TODO: explore whether DeepSpeed can have the same flow for optimizers
        # DeepSpeed did not find any optimizer in the config so they are loaded here
        *([dict(name="configure_optimizers")] if kwargs.get("plugins") == "deepspeed" else []),
        dict(name="Callback.on_pretrain_routine_start", args=(trainer, model)),
        dict(name="on_pretrain_routine_start"),
        dict(name="Callback.on_pretrain_routine_end", args=(trainer, model)),
        dict(name="on_pretrain_routine_end"),
        dict(name="Callback.on_sanity_check_start", args=(trainer, model)),
        dict(name="on_val_dataloader"),
        dict(name="val_dataloader"),
        dict(name="train", args=(False,)),
        dict(name="on_validation_model_eval"),
        dict(name="zero_grad"),
        dict(name="Callback.on_validation_start", args=(trainer, model)),
        dict(name="on_validation_start"),
        *model._eval_epoch("validation", trainer, model, val_batches, "x", device=device),
        dict(name="Callback.on_validation_end", args=(trainer, model)),
        dict(name="on_validation_end"),
        dict(name="train", args=(True,)),
        dict(name="on_validation_model_train"),
        dict(name="Callback.on_sanity_check_end", args=(trainer, model)),
        # duplicate `train` because `_run_train` calls it again in case validation wasn't run
        dict(name="train", args=(True,)),
        dict(name="on_train_dataloader"),
        dict(name="train_dataloader"),
        dict(name="Callback.on_train_start", args=(trainer, model)),
        dict(name="on_train_start"),
        dict(name="Callback.on_epoch_start", args=(trainer, model)),
        dict(name="on_epoch_start"),
        dict(name="Callback.on_train_epoch_start", args=(trainer, model)),
        dict(name="on_train_epoch_start"),
        *model._train_batch(trainer, model, train_batches, device=device, **kwargs),
        dict(name="train", args=(False,)),
        dict(name="on_validation_model_eval"),
        dict(name="zero_grad"),
        dict(name="Callback.on_validation_start", args=(trainer, model)),
        dict(name="on_validation_start"),
        *model._eval_epoch("validation", trainer, model, val_batches, "x", device=device),
        dict(name="Callback.on_validation_end", args=(trainer, model)),
        dict(name="on_validation_end"),
        dict(name="train", args=(True,)),
        dict(name="on_validation_model_train"),
        dict(name="training_epoch_end", args=([dict(loss=ANY)] * train_batches,)),
        dict(name="Callback.on_train_epoch_end", args=(trainer, model, [dict(loss=ANY)] * train_batches)),
        # `ModelCheckpoint.save_checkpoint` is called here from `Callback.on_train_epoch_end`
        dict(name="Callback.on_save_checkpoint", args=(trainer, model, saved_ckpt)),
        dict(name="on_save_checkpoint", args=(saved_ckpt,)),
        dict(name="on_train_epoch_end", args=([dict(loss=ANY)] * train_batches,)),
        dict(name="Callback.on_epoch_end", args=(trainer, model)),
        dict(name="on_epoch_end"),
        dict(name="Callback.on_train_end", args=(trainer, model)),
        dict(name="on_train_end"),
        dict(name="Callback.on_fit_end", args=(trainer, model)),
        dict(name="on_fit_end"),
        dict(name="Callback.teardown", args=(trainer, model), kwargs=dict(stage="fit")),
        dict(name="teardown", kwargs=dict(stage="fit")),
    ]
    assert called == expected


def test_trainer_model_hook_system_fit_no_val_and_resume(tmpdir):
    # initial training to get a checkpoint
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir, max_steps=1, limit_val_batches=0, progress_bar_refresh_rate=0, weights_summary=None
    )
    trainer.fit(model)
    best_model_path = trainer.checkpoint_callback.best_model_path

    # resume from checkpoint with HookedModel
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        # already performed 1 step, now resuming to do an additional 2
        max_steps=(1 + train_batches),
        limit_val_batches=0,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        resume_from_checkpoint=best_model_path,
        callbacks=[callback],
    )
    assert called == [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
    ]
    trainer.fit(model)
    saved_ckpt = {
        "callbacks": ANY,
        "epoch": 2,  # TODO: wrong saved epoch
        "global_step": (1 + train_batches),
        "lr_schedulers": ANY,
        "optimizer_states": ANY,
        "pytorch-lightning_version": __version__,
        "state_dict": ANY,
    }
    expected = [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
        dict(name="prepare_data"),
        dict(name="configure_callbacks"),
        dict(name="Callback.on_before_accelerator_backend_setup", args=(trainer, model)),
        dict(name="Callback.setup", args=(trainer, model), kwargs=dict(stage="fit")),
        dict(name="setup", kwargs=dict(stage="fit")),
        dict(
            name="on_load_checkpoint",
            args=(
                {
                    "callbacks": ANY,
                    "epoch": 1,
                    "global_step": 1,
                    "lr_schedulers": ANY,
                    "optimizer_states": ANY,
                    "pytorch-lightning_version": __version__,
                    "state_dict": ANY,
                },
            ),
        ),
        dict(name="configure_sharded_model"),
        dict(name="Callback.on_configure_sharded_model", args=(trainer, model)),
        dict(name="configure_optimizers"),
        dict(name="Callback.on_fit_start", args=(trainer, model)),
        dict(name="on_fit_start"),
        dict(name="Callback.on_pretrain_routine_start", args=(trainer, model)),
        dict(name="on_pretrain_routine_start"),
        dict(name="Callback.on_pretrain_routine_end", args=(trainer, model)),
        dict(name="on_pretrain_routine_end"),
        dict(name="train", args=(True,)),
        dict(name="on_train_dataloader"),
        dict(name="train_dataloader"),
        # even though no validation runs, we initialize the val dataloader for properties like `num_val_batches`
        dict(name="on_val_dataloader"),
        dict(name="val_dataloader"),
        dict(name="Callback.on_train_start", args=(trainer, model)),
        dict(name="on_train_start"),
        dict(name="Callback.on_epoch_start", args=(trainer, model)),
        dict(name="on_epoch_start"),
        dict(name="Callback.on_train_epoch_start", args=(trainer, model)),
        dict(name="on_train_epoch_start"),
        # TODO: wrong current epoch after reload
        *model._train_batch(trainer, model, train_batches, current_epoch=1),
        dict(name="training_epoch_end", args=([dict(loss=ANY)] * train_batches,)),
        dict(name="Callback.on_train_epoch_end", args=(trainer, model, [dict(loss=ANY)] * train_batches)),
        dict(name="Callback.on_save_checkpoint", args=(trainer, model, saved_ckpt)),
        dict(name="on_save_checkpoint", args=(saved_ckpt,)),
        dict(name="on_train_epoch_end", args=([dict(loss=ANY)] * train_batches,)),
        dict(name="Callback.on_epoch_end", args=(trainer, model)),
        dict(name="on_epoch_end"),
        dict(name="Callback.on_train_end", args=(trainer, model)),
        dict(name="on_train_end"),
        dict(name="Callback.on_fit_end", args=(trainer, model)),
        dict(name="on_fit_end"),
        dict(name="Callback.teardown", args=(trainer, model), kwargs=dict(stage="fit")),
        dict(name="teardown", kwargs=dict(stage="fit")),
    ]
    assert called == expected


@pytest.mark.parametrize("batches", (0, 2))
@pytest.mark.parametrize(
    ["verb", "noun", "dataloader", "key"], [("validate", "validation", "val", "x"), ("test", "test", "test", "y")]
)
def test_trainer_model_hook_system_eval(tmpdir, batches, verb, noun, dataloader, key):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=batches,
        limit_test_batches=batches,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        callbacks=[callback],
    )
    assert called == [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
    ]
    fn = getattr(trainer, verb)
    fn(model, verbose=False)
    hooks = [
        dict(name="train", args=(False,)),
        dict(name=f"on_{noun}_model_eval"),
        dict(name="zero_grad"),
        dict(name=f"Callback.on_{noun}_start", args=(trainer, model)),
        dict(name=f"on_{noun}_start"),
        *model._eval_epoch(noun, trainer, model, batches, key),
        dict(name=f"Callback.on_{noun}_end", args=(trainer, model)),
        dict(name=f"on_{noun}_end"),
        dict(name="train", args=(True,)),
        dict(name=f"on_{noun}_model_train"),
    ]
    expected = [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
        dict(name="prepare_data"),
        dict(name="configure_callbacks"),
        dict(name="Callback.on_before_accelerator_backend_setup", args=(trainer, model)),
        dict(name="Callback.setup", args=(trainer, model), kwargs=dict(stage=verb)),
        dict(name="setup", kwargs=dict(stage=verb)),
        dict(name="configure_sharded_model"),
        dict(name="Callback.on_configure_sharded_model", args=(trainer, model)),
        dict(name=f"on_{dataloader}_dataloader"),
        dict(name=f"{dataloader}_dataloader"),
        *(hooks if batches else []),
        dict(name="Callback.teardown", args=(trainer, model), kwargs=dict(stage=verb)),
        dict(name="teardown", kwargs=dict(stage=verb)),
    ]
    assert called == expected


def test_trainer_model_hook_system_predict(tmpdir):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir, limit_predict_batches=batches, progress_bar_refresh_rate=0, callbacks=[callback]
    )
    assert called == [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
    ]
    trainer.predict(model)
    expected = [
        dict(name="Callback.on_init_start", args=(trainer,)),
        dict(name="Callback.on_init_end", args=(trainer,)),
        dict(name="prepare_data"),
        dict(name="configure_callbacks"),
        dict(name="Callback.on_before_accelerator_backend_setup", args=(trainer, model)),
        dict(name="Callback.setup", args=(trainer, model), kwargs=dict(stage="predict")),
        dict(name="setup", kwargs=dict(stage="predict")),
        dict(name="configure_sharded_model"),
        dict(name="Callback.on_configure_sharded_model", args=(trainer, model)),
        dict(name="on_predict_dataloader"),
        dict(name="predict_dataloader"),
        dict(name="train", args=(False,)),
        dict(name="on_predict_model_eval"),
        dict(name="zero_grad"),
        dict(name="Callback.on_predict_start", args=(trainer, model)),
        dict(name="on_predict_start"),
        # TODO: `{,Callback}.on_epoch_{start,end}`
        dict(name="Callback.on_predict_epoch_start", args=(trainer, model)),
        dict(name="on_predict_epoch_start"),
        *model._predict_batch(trainer, model, batches),
        # TODO: `predict_epoch_end`
        dict(name="Callback.on_predict_epoch_end", args=(trainer, model, [[ANY] * batches])),
        dict(name="on_predict_epoch_end", args=([[ANY] * batches],)),
        dict(name="Callback.on_predict_end", args=(trainer, model)),
        dict(name="on_predict_end"),
        # TODO: `on_predict_model_train`
        dict(name="Callback.teardown", args=(trainer, model), kwargs=dict(stage="predict")),
        dict(name="teardown", kwargs=dict(stage="predict")),
    ]
    assert called == expected


# TODO: add test for tune


def test_hooks_with_different_argument_names(tmpdir):
    """
    Test that argument names can be anything in the hooks
    """

    class CustomBoringModel(BoringModel):
        def assert_args(self, x, batch_nb):
            assert isinstance(x, torch.Tensor)
            assert x.size() == (1, 32)
            assert isinstance(batch_nb, int)

        def training_step(self, x1, batch_nb1):
            self.assert_args(x1, batch_nb1)
            return super().training_step(x1, batch_nb1)

        def validation_step(self, x2, batch_nb2):
            self.assert_args(x2, batch_nb2)
            return super().validation_step(x2, batch_nb2)

        def test_step(self, x3, batch_nb3, dl_idx3):
            self.assert_args(x3, batch_nb3)
            assert isinstance(dl_idx3, int)
            return super().test_step(x3, batch_nb3)

        def predict(self, x4, batch_nb4, dl_idx4):
            self.assert_args(x4, batch_nb4)
            assert isinstance(dl_idx4, int)
            return super().predict(x4, batch_nb4, dl_idx4)

        def test_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

        def predict_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

    model = CustomBoringModel()
    model.test_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=5)

    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    trainer.test(ckpt_path=None)

    preds = trainer.predict(model)
    assert len(preds) == 2
    assert all(len(x) == 5 for x in preds)


def test_trainer_datamodule_hook_system(tmpdir):
    """Test the LightningDataModule hook system."""

    class HookedDataModule(BoringDataModule):
        def __init__(self, called):
            super().__init__()

            def call(hook, fn, *args, **kwargs):
                out = fn(*args, **kwargs)
                d = {"name": hook}
                if args:
                    d["args"] = args
                if kwargs:
                    d["kwargs"] = kwargs
                called.append(d)
                return out

            for h in get_members(LightningDataModule):
                attr = getattr(self, h)
                setattr(self, h, partial(call, h, attr))

    model = BoringModel()
    batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=batches,
        limit_val_batches=batches,
        limit_test_batches=batches,
        limit_predict_batches=batches,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        reload_dataloaders_every_epoch=True,
    )

    called = []
    dm = HookedDataModule(called)
    trainer.fit(model, datamodule=dm)
    batch_transfer = [
        dict(name="on_before_batch_transfer", args=(ANY, 0)),
        dict(name="transfer_batch_to_device", args=(ANY, torch.device("cpu"), 0)),
        dict(name="on_after_batch_transfer", args=(ANY, 0)),
    ]
    expected = [
        dict(name="prepare_data"),
        dict(name="setup", kwargs=dict(stage="fit")),
        dict(name="val_dataloader"),
        *batch_transfer * batches,
        dict(name="train_dataloader"),
        *batch_transfer * batches,
        dict(name="val_dataloader"),
        *batch_transfer * batches,
        dict(
            name="on_save_checkpoint",
            args=(
                {
                    "callbacks": ANY,
                    "epoch": 1,
                    "global_step": 2,
                    "lr_schedulers": ANY,
                    "optimizer_states": ANY,
                    "pytorch-lightning_version": __version__,
                    "state_dict": ANY,
                },
            ),
        ),
        dict(name="teardown", kwargs=dict(stage="fit")),
    ]
    assert called == expected

    called = []
    dm = HookedDataModule(called)
    trainer.validate(model, datamodule=dm, verbose=False)
    expected = [
        dict(name="prepare_data"),
        dict(name="setup", kwargs=dict(stage="validate")),
        dict(name="val_dataloader"),
        *batch_transfer * batches,
        dict(name="teardown", kwargs=dict(stage="validate")),
    ]
    assert called == expected

    called = []
    dm = HookedDataModule(called)
    trainer.test(model, datamodule=dm, verbose=False)
    expected = [
        dict(name="prepare_data"),
        dict(name="setup", kwargs=dict(stage="test")),
        dict(name="test_dataloader"),
        *batch_transfer * batches,
        dict(name="teardown", kwargs=dict(stage="test")),
    ]
    assert called == expected

    called = []
    dm = HookedDataModule(called)
    trainer.predict(model, datamodule=dm)
    expected = [
        dict(name="prepare_data"),
        dict(name="setup", kwargs=dict(stage="predict")),
        dict(name="predict_dataloader"),
        *batch_transfer * batches,
        dict(name="teardown", kwargs=dict(stage="predict")),
    ]
    assert called == expected
