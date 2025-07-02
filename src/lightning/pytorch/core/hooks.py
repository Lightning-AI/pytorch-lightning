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
"""Various hooks to be used in the Lightning code."""

from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from lightning.pytorch.utilities import move_data_to_device
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS


class ModelHooks:
    """Hooks to be used in LightningModule."""

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.

        If on DDP it is called on every process

        """

    def on_fit_end(self) -> None:
        """Called at the very end of fit.

        If on DDP it is called on every process

        """

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""

    def on_train_end(self) -> None:
        """Called at the end of training before logger experiment is closed."""

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""

    def on_validation_end(self) -> None:
        """Called at the end of validation."""

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""

    def on_test_end(self) -> None:
        """Called at the end of testing."""

    def on_predict_start(self) -> None:
        """Called at the beginning of predicting."""

    def on_predict_end(self) -> None:
        """Called at the end of predicting."""

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        """Called in the training loop before anything happens for that batch.

        If you return -1 here, you will skip training for the rest of the current epoch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch

        """

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called in the training loop after the batch.

        Args:
            outputs: The outputs of training_step(x)
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called in the validation loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the validation DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called in the validation loop after the batch.

        Args:
            outputs: The outputs of validation_step(x)
            batch: The batched data as it is returned by the validation DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called in the test loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called in the test loop after the batch.

        Args:
            outputs: The outputs of test_step(x)
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called in the predict loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called in the predict loop after the batch.

        Args:
            outputs: The outputs of predict_step(x)
            batch: The batched data as it is returned by the prediction DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_validation_model_zero_grad(self) -> None:
        """Called by the training loop to release gradients before entering the validation loop."""
        self.zero_grad()

    def on_validation_model_eval(self) -> None:
        """Called when the validation loop starts.

        The validation loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
        to change the behavior. See also :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_validation_model_train`.

        """
        self.trainer.model.eval()

    def on_validation_model_train(self) -> None:
        """Called when the validation loop ends.

        The validation loop by default restores the `training` mode of the LightningModule to what it was before
        starting validation. Override this hook to change the behavior. See also
        :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_validation_model_eval`.

        """
        # The loop won't call this hook unless it is overridden. The line below is here in case the user calls super().
        self.trainer.model.train()

    def on_test_model_eval(self) -> None:
        """Called when the test loop starts.

        The test loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
        to change the behavior. See also :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_test_model_train`.

        """
        self.trainer.model.eval()

    def on_test_model_train(self) -> None:
        """Called when the test loop ends.

        The test loop by default restores the `training` mode of the LightningModule to what it was before
        starting testing. Override this hook to change the behavior. See also
        :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_test_model_eval`.

        """
        # The loop won't call this hook unless it is overridden. The line below is here in case the user calls super().
        self.trainer.model.train()

    def on_predict_model_eval(self) -> None:
        """Called when the predict loop starts.

        The predict loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
        to change the behavior.

        """
        self.trainer.model.eval()

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch.

        To access all batch outputs at the end of the epoch, you can cache step outputs as an attribute of the
        :class:`~lightning.pytorch.LightningModule` and access them in this hook:

        .. code-block:: python

            class MyLightningModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.training_step_outputs = []

                def training_step(self):
                    loss = ...
                    self.training_step_outputs.append(loss)
                    return loss

                def on_train_epoch_end(self):
                    # do something with all training_step outputs, for example:
                    epoch_mean = torch.stack(self.training_step_outputs).mean()
                    self.log("training_epoch_mean", epoch_mean)
                    # free up the memory
                    self.training_step_outputs.clear()

        """

    def on_validation_epoch_start(self) -> None:
        """Called in the validation loop at the very beginning of the epoch."""

    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""

    def on_test_epoch_start(self) -> None:
        """Called in the test loop at the very beginning of the epoch."""

    def on_test_epoch_end(self) -> None:
        """Called in the test loop at the very end of the epoch."""

    def on_predict_epoch_start(self) -> None:
        """Called at the beginning of predicting."""

    def on_predict_epoch_end(self) -> None:
        """Called at the end of predicting."""

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """Called after ``training_step()`` and before ``optimizer.zero_grad()``.

        Called in the training loop after taking an optimizer step and before zeroing grads.
        Good place to inspect weight information with weights updated.

        This is where it is called::

            for optimizer in optimizers:
                out = training_step(...)

                model.on_before_zero_grad(optimizer) # < ---- called here
                optimizer.zero_grad()

                backward()

        Args:
            optimizer: The optimizer for which grads should be zeroed.

        """

    def on_before_backward(self, loss: Tensor) -> None:
        """Called before ``loss.backward()``.

        Args:
            loss: Loss divided by number of batches for gradient accumulation and scaled if using AMP.

        """
        pass

    def on_after_backward(self) -> None:
        """Called after ``loss.backward()`` and before optimizers are stepped.

        Note:
            If using native AMP, the gradients will not be unscaled at this point.
            Use the ``on_before_optimizer_step`` if you need the unscaled gradients.

        """

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Called before ``optimizer.step()``.

        If using gradient accumulation, the hook is called once the gradients have been accumulated.
        See: :paramref:`~lightning.pytorch.trainer.trainer.Trainer.accumulate_grad_batches`.

        If using AMP, the loss will be unscaled before calling this hook.
        See these `docs <https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients>`__
        for more information on the scaling of gradients.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            optimizer: Current optimizer being used.

        Example::

            def on_before_optimizer_step(self, optimizer):
                # example to inspect gradient information in tensorboard
                if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
                    for k, v in self.named_parameters():
                        self.logger.experiment.add_histogram(
                            tag=k, values=v.grad, global_step=self.trainer.global_step
                        )

        """

    def configure_sharded_model(self) -> None:
        """Deprecated.

        Use :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` instead.

        """

    def configure_model(self) -> None:
        """Hook to create modules in a strategy and precision aware context.

        This is particularly useful for when using sharded strategies (FSDP and DeepSpeed), where we'd like to shard
        the model instantly to save memory and initialization time.
        For non-sharded strategies, you can choose to override this hook or to initialize your model under the
        :meth:`~lightning.pytorch.trainer.trainer.Trainer.init_module` context manager.

        This hook is called during each of fit/val/test/predict stages in the same process, so ensure that
        implementation of this hook is **idempotent**, i.e., after the first time the hook is called, subsequent calls
        to it should be a no-op.

        """


class DataHooks:
    """Hooks to be used for data related stuff."""

    def __init__(self) -> None:
        """
        Attributes:
            prepare_data_per_node:
                If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data.
            allow_zero_length_dataloader_with_multiple_devices:
                If True, dataloader with zero length within local rank is allowed.
                Default value is False.
        """
        super().__init__()
        self.prepare_data_per_node: bool = True
        self.allow_zero_length_dataloader_with_multiple_devices: bool = False

    def prepare_data(self) -> None:
        """Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single process,
        so you can safely add your downloading logic within.

        .. warning:: DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In a distributed environment, ``prepare_data`` can be called in two ways
        (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = True


            # call on GLOBAL_RANK=0 (great for shared file systems)
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = False

        This is called before requesting the dataloaders:

        .. code-block:: python

            model.prepare_data()
            initialize_distributed()
            model.setup(stage)
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()
            model.predict_dataloader()

        """

    def setup(self, stage: str) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when you
        need to build models dynamically or adjust something about them. This hook is called on every process when
        using DDP.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        Example::

            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)

        """

    def teardown(self, stage: str) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        """

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """An iterable or collection of iterables specifying training samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        """
        raise MisconfigurationException("`train_dataloader` must be implemented to be used with the Lightning Trainer")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        r"""An iterable or collection of iterables specifying test samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data


        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.test`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.

        """
        raise MisconfigurationException("`test_dataloader` must be implemented to be used with the Lightning Trainer")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        r"""An iterable or collection of iterables specifying validation samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`
        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.

        """
        raise MisconfigurationException("`val_dataloader` must be implemented to be used with the Lightning Trainer")

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        r"""An iterable or collection of iterables specifying prediction samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.

        """
        raise MisconfigurationException(
            "`predict_dataloader` must be implemented to be used with the Lightning Trainer"
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors wrapped in a custom data
        structure.

        The data types listed below (and any arbitrary nesting of them) are supported out of the box:

        - :class:`torch.Tensor` or anything that implements `.to(...)`
        - :class:`list`
        - :class:`dict`
        - :class:`tuple`

        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

        Note:
            This hook should only transfer the data and not modify it, nor should it move the data to
            any other device than the one passed in as argument (unless you know what you are doing).
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.

        Example::

            def transfer_batch_to_device(self, batch, device, dataloader_idx):
                if isinstance(batch, CustomBatch):
                    # move all tensors in your custom data structure to the device
                    batch.samples = batch.samples.to(device)
                    batch.targets = batch.targets.to(device)
                elif dataloader_idx == 0:
                    # skip device transfer for the first dataloader or anything you wish
                    pass
                else:
                    batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
                return batch

        See Also:
            - :meth:`move_data_to_device`
            - :meth:`apply_to_collection`

        """
        return move_data_to_device(batch, device)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch before it is transferred to the device.

        Note:
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data

        Example::

            def on_before_batch_transfer(self, batch, dataloader_idx):
                batch['x'] = transforms(batch['x'])
                return batch

        See Also:
            - :meth:`on_after_batch_transfer`
            - :meth:`transfer_batch_to_device`

        """
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch after it is transferred to the device.

        Note:
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data

        Example::

            def on_after_batch_transfer(self, batch, dataloader_idx):
                batch['x'] = gpu_transforms(batch['x'])
                return batch

        See Also:
            - :meth:`on_before_batch_transfer`
            - :meth:`transfer_batch_to_device`

        """
        return batch


class CheckpointHooks:
    """Hooks to be used with Checkpointing."""

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        r"""Called by Lightning to restore your model. If you saved something with :meth:`on_save_checkpoint` this is
        your chance to restore this.

        Args:
            checkpoint: Loaded checkpoint

        Example::

            def on_load_checkpoint(self, checkpoint):
                # 99% of the time you don't need to implement this method
                self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']

        Note:
            Lightning auto-restores global step, epoch, and train state including amp scaling.
            There is no need for you to restore anything regarding training.

        """

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        r"""Called by Lightning when saving a checkpoint to give you a chance to store anything else you might want to
        save.

        Args:
            checkpoint: The full checkpoint dictionary before it gets dumped to a file.
                Implementations of this hook can insert additional data into this dictionary.

        Example::

            def on_save_checkpoint(self, checkpoint):
                # 99% of use cases you don't need to implement this method
                checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

        Note:
            Lightning saves all aspects of training (epoch, global step, etc...)
            including amp scaling.
            There is no need for you to store anything about training.

        """
