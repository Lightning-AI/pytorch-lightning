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
"""Various hooks to be used in the Lightning code."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.utilities import move_data_to_device, rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ModelHooks:
    """Hooks to be used in LightningModule."""

    def on_fit_start(self) -> None:
        """
        Called at the very beginning of fit.
        If on DDP it is called on every process
        """

    def on_fit_end(self) -> None:
        """
        Called at the very end of fit.
        If on DDP it is called on every process
        """

    def on_train_start(self) -> None:
        """
        Called at the beginning of training after sanity check.
        """

    def on_train_end(self) -> None:
        """
        Called at the end of training before logger experiment is closed.
        """

    def on_validation_start(self) -> None:
        """
        Called at the beginning of validation.
        """

    def on_validation_end(self) -> None:
        """
        Called at the end of validation.
        """

    def on_test_start(self) -> None:
        """
        Called at the beginning of testing.
        """

    def on_test_end(self) -> None:
        """
        Called at the end of testing.
        """

    def on_predict_start(self) -> None:
        """
        Called at the beginning of predicting.
        """

    def on_predict_end(self) -> None:
        """
        Called at the end of predicting.
        """

    def on_pretrain_routine_start(self) -> None:
        """
        Called at the beginning of the pretrain routine (between fit and train start).

        - fit
        - pretrain_routine start
        - pretrain_routine end
        - training_start

        """

    def on_pretrain_routine_end(self) -> None:
        """
        Called at the end of the pretrain routine (between fit and train start).

        - fit
        - pretrain_routine start
        - pretrain_routine end
        - training_start

        """

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the training loop before anything happens for that batch.

        If you return -1 here, you will skip training for the rest of the current epoch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the training loop after the batch.

        Args:
            outputs: The outputs of training_step_end(training_step(x))
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the validation loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the validation DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_validation_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Called in the validation loop after the batch.

        Args:
            outputs: The outputs of validation_step_end(validation_step(x))
            batch: The batched data as it is returned by the validation DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the test loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_test_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Called in the test loop after the batch.

        Args:
            outputs: The outputs of test_step_end(test_step(x))
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the predict loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the predict loop after the batch.

        Args:
            outputs: The outputs of predict_step_end(test_step(x))
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """

    def on_validation_model_eval(self) -> None:
        """
        Sets the model to eval during the val loop
        """
        self.trainer.model.eval()

    def on_validation_model_train(self) -> None:
        """
        Sets the model to train during the val loop
        """
        self.trainer.model.train()

    def on_test_model_train(self) -> None:
        """
        Sets the model to train during the test loop
        """
        self.trainer.model.train()

    def on_test_model_eval(self) -> None:
        """
        Sets the model to eval during the test loop
        """
        self.trainer.model.eval()

    def on_predict_model_eval(self) -> None:
        """
        Sets the model to eval during the predict loop
        """
        self.trainer.model.eval()

    def on_epoch_start(self) -> None:
        """
        Called when either of train/val/test epoch begins.
        """

    def on_epoch_end(self) -> None:
        """
        Called when either of train/val/test epoch ends.
        """

    def on_train_epoch_start(self) -> None:
        """
        Called in the training loop at the very beginning of the epoch.
        """

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        """
        Called in the training loop at the very end of the epoch.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the LightningModule OR
        2. Cache data across steps on the attribute(s) of the `LightningModule` and access them in this hook
        """

    def on_validation_epoch_start(self) -> None:
        """
        Called in the validation loop at the very beginning of the epoch.
        """

    def on_validation_epoch_end(self) -> None:
        """
        Called in the validation loop at the very end of the epoch.
        """

    def on_test_epoch_start(self) -> None:
        """
        Called in the test loop at the very beginning of the epoch.
        """

    def on_test_epoch_end(self) -> None:
        """
        Called in the test loop at the very end of the epoch.
        """

    def on_predict_epoch_start(self) -> None:
        """
        Called at the beginning of predicting.
        """

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        """
        Called at the end of predicting.
        """

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """
        Called after ``training_step()`` and before ``optimizer.zero_grad()``.

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

    def on_after_backward(self) -> None:
        """
        Called in the training loop after loss.backward() and before optimizers do anything.
        This is the ideal place to inspect or log gradient information.

        Example::

            def on_after_backward(self):
                # example to inspect gradient information in tensorboard
                if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
                    for k, v in self.named_parameters():
                        self.logger.experiment.add_histogram(
                            tag=k, values=v.grad, global_step=self.trainer.global_step
                        )

        """

    def on_post_move_to_device(self) -> None:
        """
        Called in the ``parameter_validation`` decorator after :meth:`~pytorch_lightning.core.LightningModule.to`
        is called. This is a good place to tie weights between modules after moving them to a device. Can be
        used when training models with weight sharing properties on TPU.

        Addresses the handling of shared weights on TPU:
        https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks

        Example::

            def on_post_move_to_device(self):
                self.decoder.weight = self.encoder.weight

        """

    def configure_sharded_model(self) -> None:
        """
        Hook to create modules in a distributed aware context. This is useful for when using sharded plugins,
        where we'd like to shard the model instantly, which is useful for extremely large models
        which can save memory and initialization time.

        The accelerator manages whether to call this hook at every given stage.
        For sharded plugins where model parallelism is required, the hook is usually on called once
        to initialize the sharded parameters, and not called again in the same process.

        By default for accelerators/plugins that do not use model sharding techniques,
        this hook is called during each fit/val/test/predict stages.
        """


class DataHooks:
    """Hooks to be used for data related stuff."""

    def prepare_data(self) -> None:
        """
        Use this to download and prepare data.

        .. warning:: DO NOT set state to the model (use `setup` instead)
            since this is NOT called on every GPU in DDP/TPU

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In DDP prepare_data can be called in two ways (using Trainer(prepare_data_per_node)):

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            Trainer(prepare_data_per_node=True)

            # call on GLOBAL_RANK=0 (great for shared file systems)
            Trainer(prepare_data_per_node=False)

        This is called before requesting the dataloaders:

        .. code-block:: python

            model.prepare_data()
                if ddp/tpu: init()
            model.setup(stage)
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called at the beginning of fit (train + validate), validate, test, and predict.
        This is a good hook when you need to build models dynamically or adjust something about them.
        This hook is called on every process when using DDP.

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

                def setup(stage):
                    data = Load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)

        """

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Called at the end of fit (train + validate), validate, test, predict, or tune.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Implement one or more PyTorch DataLoaders for training.

        Return:
            Either a single PyTorch :class:`~torch.utils.data.DataLoader` or a collection of these
            (list, dict, nested lists and dicts). In the case of multiple dataloaders, please see
            this :ref:`page <multiple-training-dataloaders>`

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`setup`
        - :meth:`train_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Example::

            # single dataloader
            def train_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                return loader

            # multiple dataloaders, return as list
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a list of tensors: [batch_mnist, batch_cifar]
                return [mnist_loader, cifar_loader]

            # multiple dataloader, return as dict
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a dict of tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
                return {'mnist': mnist_loader, 'cifar': cifar_loader}

        """
        rank_zero_warn("`train_dataloader` must be implemented to be used with the Lightning Trainer")

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement one or multiple PyTorch DataLoaders for testing.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data


        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`setup`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Return:
            Single or multiple PyTorch DataLoaders.

        Example::

            def test_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )

                return loader

            # can also return multiple dataloaders
            def test_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.

        Note:
            In the case where you return multiple test dataloaders, the :meth:`test_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement one or multiple PyTorch DataLoaders for validation.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            Single or multiple PyTorch DataLoaders.

        Examples::

            def val_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False,
                                transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )

                return loader

            # can also return multiple dataloaders
            def val_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.

        Note:
            In the case where you return multiple validation dataloaders, the :meth:`validation_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement one or multiple PyTorch DataLoaders for prediction.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            Single or multiple PyTorch DataLoaders.

        Note:
            In the case where you return multiple prediction dataloaders, the :meth:`predict`
            will have an argument ``dataloader_idx`` which matches the order here.
        """

    def on_train_dataloader(self) -> None:
        """Called before requesting the train dataloader."""

    def on_val_dataloader(self) -> None:
        """Called before requesting the val dataloader."""

    def on_test_dataloader(self) -> None:
        """Called before requesting the test dataloader."""

    def on_predict_dataloader(self) -> None:
        """Called before requesting the predict dataloader."""

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """
        Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors
        wrapped in a custom data structure.

        The data types listed below (and any arbitrary nesting of them) are supported out of the box:

        - :class:`torch.Tensor` or anything that implements `.to(...)`
        - :class:`list`
        - :class:`dict`
        - :class:`tuple`
        - :class:`torchtext.data.batch.Batch`

        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

        Note:
            This hook should only transfer the data and not modify it, nor should it move the data to
            any other device than the one passed in as argument (unless you know what you are doing).
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Note:
            This hook only runs on single GPU training and DDP (no data-parallel).
            Data-Parallel support will come in near future.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.

        Example::

            def transfer_batch_to_device(self, batch, device):
                if isinstance(batch, CustomBatch):
                    # move all tensors in your custom data structure to the device
                    batch.samples = batch.samples.to(device)
                    batch.targets = batch.targets.to(device)
                else:
                    batch = super().transfer_batch_to_device(data, device)
                return batch

        Raises:
            MisconfigurationException:
                If using data-parallel, ``Trainer(accelerator='dp')``.

        See Also:
            - :meth:`move_data_to_device`
            - :meth:`apply_to_collection`
        """
        return move_data_to_device(batch, device)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        Override to alter or apply batch augmentations to your batch before it is transferred to the device.

        Note:
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Note:
            This hook only runs on single GPU training and DDP (no data-parallel).
            Data-Parallel support will come in near future.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data

        Example::

            def on_before_batch_transfer(self, batch, dataloader_idx):
                batch['x'] = transforms(batch['x'])
                return batch

        Raises:
            MisconfigurationException:
                If using data-parallel, ``Trainer(accelerator='dp')``.

        See Also:
            - :meth:`on_after_batch_transfer`
            - :meth:`transfer_batch_to_device`
        """
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        Override to alter or apply batch augmentations to your batch after it is transferred to the device.

        Note:
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Note:
            This hook only runs on single GPU training and DDP (no data-parallel).
            Data-Parallel support will come in near future.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data

        Example::

            def on_after_batch_transfer(self, batch, dataloader_idx):
                batch['x'] = gpu_transforms(batch['x'])
                return batch

        Raises:
            MisconfigurationException:
                If using data-parallel, ``Trainer(accelerator='dp')``.

        See Also:
            - :meth:`on_before_batch_transfer`
            - :meth:`transfer_batch_to_device`
        """
        return batch


class CheckpointHooks:
    """Hooks to be used with Checkpointing."""

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""
        Called by Lightning to restore your model.
        If you saved something with :meth:`on_save_checkpoint` this is your chance to restore this.

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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""
        Called by Lightning when saving a checkpoint to give you a chance to store anything
        else you might want to save.

        Args:
            checkpoint: Checkpoint to be saved

        Example::

            def on_save_checkpoint(self, checkpoint):
                # 99% of use cases you don't need to implement this method
                checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

        Note:
            Lightning saves all aspects of training (epoch, global step, etc...)
            including amp scaling.
            There is no need for you to store anything about training.

        """
