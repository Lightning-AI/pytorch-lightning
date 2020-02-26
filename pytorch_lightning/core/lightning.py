import collections
import inspect
import logging as log
import os
import warnings
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Union, Tuple, List, Optional, Callable, Dict

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.core.decorators import data_loader
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.core.saving import ModelIO, load_hparams_from_tags_csv
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True

except ImportError:
    XLA_AVAILABLE = False


class LightningModule(ABC, GradInformation, ModelIO, ModelHooks):

    def __init__(self, *args, **kwargs):
        super(LightningModule, self).__init__(*args, **kwargs)

        #: Current dtype
        self.dtype = torch.FloatTensor

        self.exp_save_path = None

        #: The current epoch
        self.current_epoch = 0

        #: Total training batches seen across all epochs
        self.global_step = 0

        self.loaded_optimizer_states_dict = {}

        #: Pointer to the trainer object
        self.trainer = None

        #: Pointer to the logger object
        self.logger = None
        self.example_input_array = None

        #: True if your model is currently running on GPUs.
        #: Useful to set flags around the LightningModule for different CPU vs GPU behavior.
        self.on_gpu = False

        #: True if using dp
        self.use_dp = False

        #: True if using ddp
        self.use_ddp = False

        #: True if using ddp2
        self.use_ddp2 = False

        #: True if using amp
        self.use_amp = False

    def print(self, *args, **kwargs) -> None:
        r"""
        Prints only from process 0. Use this in any distributed mode to log only once

        Args:
            x (object): The thing to print

        Example
        -------

        .. code-block:: python

            # example if we were using this model as a feature extractor
            def forward(self, x):
                self.print(x, 'in loader')

        """
        if self.trainer.proc_rank == 0:
            log.info(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        r"""
        Same as torch.nn.Module.forward(), however in Lightning you want this to define
        the  operations you want to use for prediction (ie: on a server or as a feature extractor).

        Normally you'd call self.forward() from your training_step() method. This makes it easy to write a complex
        system for training with the outputs you'd want in a prediction setting.

        Args:
            x (tensor): Whatever  you decide to define in the forward method

        Return:
            Predicted output

        Example
        -------

        .. code-block:: python

            # example if we were using this model as a feature extractor
            def forward(self, x):
                feature_maps = self.convnet(x)
                return feature_maps

            def training_step(self, batch, batch_idx):
                x, y = batch
                feature_maps = self.forward(x)
                logits = self.classifier(feature_maps)

                # ...
                return loss

            # splitting it this way allows model to be used a feature extractor
            model = MyModelAbove()

            inputs = server.get_request()
            results = model(inputs)
            server.write_results(results)

            # -------------
            # This is in stark contrast to torch.nn.Module where normally you would have this:
            def forward(self, batch):
                x, y = batch
                feature_maps = self.convnet(x)
                logits = self.classifier(feature_maps)
                return logits

        """

    def training_step(self, *args, **kwargs) -> dict:
        r"""return loss, dict with metrics for tqdm

        Args:
            batch (torch.nn.Tensor | (Tensor, Tensor) | [Tensor, Tensor]): The output of your dataloader.
                A tensor, tuple or list
            batch_idx (int): Integer displaying index of this batch
            optimizer_idx (int): If using multiple optimizers, this argument will also be present.
            hiddens(:`Tensor <https://pytorch.org/docs/stable/tensors.html>`_): Passed in if truncated_bptt_steps > 0.

        :param

        :return: dict with loss key and optional log, progress keys
         if implementing training_step, return whatever you need in that step:

            - loss -> tensor scalar [REQUIRED]
            - progress_bar -> Dict for progress bar display. Must have only tensors
            - log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)

        In this step you'd normally do the forward pass and calculate the loss for a batch.
         You can also do fancier things like multiple forward passes or something specific to your model.

        Example
        -------

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                x, y, z = batch

                # implement your own
                out = self.forward(x)
                loss = self.loss(out, x)

                logger_logs = {'training_loss': loss} # optional (MUST ALL BE TENSORS)

                # if using TestTubeLogger or TensorBoardLogger you can nest scalars
                logger_logs = {'losses': logger_logs} # optional (MUST ALL BE TENSORS)

                output = {
                    'loss': loss, # required
                    'progress_bar': {'training_loss': loss}, # optional (MUST ALL BE TENSORS)
                    'log': logger_logs
                }

                # return a dict
                return output

        If you define multiple optimizers, this step will also be called with an additional `optimizer_idx` param.

        .. code-block:: python

            # Multiple optimizers (ie: GANs)
            def training_step(self, batch, batch_idx, optimizer_idx):
                if optimizer_idx == 0:
                    # do training_step with encoder
                if optimizer_idx == 1:
                    # do training_step with decoder


        If you add truncated back propagation through time you will also get an additional
         argument with the hidden states of the previous step.

        .. code-block:: python

            # Truncated back-propagation through time
            def training_step(self, batch, batch_idx, hiddens):
                # hiddens are the hiddens from the previous truncated backprop step
                ...
                out, hiddens = self.lstm(data, hiddens)
                ...

                return {
                    "loss": ...,
                    "hiddens": hiddens  # remember to detach() this
                }

        You can also return a -1 instead of a dict to stop the current loop. This is useful
         if you want to break out of the current training epoch early.
        """

    def training_end(self, outputs: dict) -> dict:
        """return loss, dict with metrics for tqdm

        :param outputs: What you return in `training_step`.
        :return: Dictionary with loss key and optional log, progress keys:
            - loss -> tensor scalar [REQUIRED]
            - progress_bar -> Dict for progress bar display. Must have only tensors
            - log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)

        In certain cases (dp, ddp2), you might want to use all outputs of every process to do something.
        For instance, if using negative samples, you could run a batch via dp and use ALL the outputs
        for a single softmax across the full batch (ie: the denominator would use the full batch).

        In this case you should define training_end to perform those calculations.

        Example
        -------

        .. code-block:: python

            # WITHOUT training_end
            # if used in DP or DDP2, this batch is 1/num_gpus large
            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self.forward(x)
                loss = self.softmax(out)
                loss = nce_loss(loss)
                return {'loss': loss}

            # --------------
            # with training_end to do softmax over the full batch
            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self.forward(x)
                return {'out': out}

            def training_end(self, outputs):
                # this out is now the full size of the batch
                out = outputs['out']

                # this softmax now uses the full batch size
                loss = self.softmax(out)
                loss = nce_loss(loss)
                return {'loss': loss}

        If you define multiple optimizers, this step will also be called with an additional `optimizer_idx` param.

        .. code-block:: python

            # Multiple optimizers (ie: GANs)
            def training_step(self, batch, batch_idx, optimizer_idx):
                if optimizer_idx == 0:
                    # do training_step with encoder
                if optimizer_idx == 1:
                    # do training_step with decoder

        If you add truncated back propagation through time you will also get an additional argument
         with the hidden states of the previous step.

        .. code-block:: python

            # Truncated back-propagation through time
            def training_step(self, batch, batch_idx, hiddens):
                # hiddens are the hiddens from the previous truncated backprop step

        You can also return a -1 instead of a dict to stop the current loop. This is useful if you want to
        break out of the current training epoch early.
        """

    def validation_step(self, *args, **kwargs) -> dict:
        r"""

        This is the validation loop. It is called for each batch of the validation set.
        Whatever is returned from here will be passed in as a list on validation_end.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.

        Args:
            batch (torch.nn.Tensor | (Tensor, Tensor) | [Tensor, Tensor]): The output of your dataloader.
                A tensor, tuple or list
            batch_idx (int): The index of this batch
            dataloader_idx (int): The index of the dataloader that produced this batch (only if multiple
                val datasets used)

        Return:
            Dict or OrderedDict - passed to the validation_end step

        .. code-block:: python

            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx)

            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idxdx)

        Example
        -------

        .. code-block:: python

            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self.forward(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # all optional...
                # return whatever you need for the collation function validation_end
                output = OrderedDict({
                    'val_loss': loss_val,
                    'val_acc': torch.tensor(val_acc), # everything must be a tensor
                })

                # return an optional dict
                return output

        If you pass in multiple validation datasets, validation_step will have an additional argument.

        .. code-block:: python

            # CASE 2: multiple validation datasets
            def validation_step(self, batch, batch_idx, dataset_idx):
                # dataset_idx tells you which dataset this is.

        .. note:: If you don't need to validate you don't need to implement this method.

        .. note:: When the validation_step is called, the model has been put in eval mode and PyTorch gradients
            have been disabled. At the end of validation, model goes back to training mode and gradients are enabled.
        """

    def test_step(self, *args, **kwargs) -> dict:
        """return whatever outputs will need to be aggregated in test_end
        :param batch: The output of your dataloader. A tensor, tuple or list
        :param int batch_idx: Integer displaying which batch this is
        :param int dataloader_idx: Integer displaying which dataloader this is (only if multiple test datasets used)
        :return dict: Dict or OrderedDict with metrics to display in progress bar. All keys must be tensors.

        .. code-block:: python

            # if you have one test dataloader:
            def test_step(self, batch, batch_idx)

            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idxdx)


        **OPTIONAL**
        If you don't need to test you don't need to implement this method.
        In this step you'd normally generate examples or
        calculate anything of interest such as accuracy.

        When the validation_step is called, the model has been put in eval mode
        and PyTorch gradients have been disabled.
        At the end of validation, model goes back to training mode and gradients are enabled.

        The dict you return here will be available in the `test_end` method.

        This function is used when you execute `trainer.test()`.

        Example
        -------

        .. code-block:: python

            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self.forward(x)
                loss = self.loss(out, y)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # all optional...
                # return whatever you need for the collation function test_end
                output = OrderedDict({
                    'test_loss': loss_test,
                    'test_acc': torch.tensor(test_acc), # everything must be a tensor
                })

                # return an optional dict
                return output


        If you pass in multiple test datasets, `test_step` will have an additional argument.

        .. code-block:: python

            # CASE 2: multiple test datasets
            def test_step(self, batch, batch_idx, dataset_idx):
                # dataset_idx tells you which dataset this is.


        The `dataset_idx` corresponds to the order of datasets returned in `test_dataloader`.
        """

    def validation_end(self, outputs: list) -> dict:
        """Outputs has the appended output after each validation step.

        :param outputs: List of outputs you defined in validation_step, or if there are multiple dataloaders,
         a list containing a list of outputs for each dataloader
        :return: Dictionary or OrderedDict with optional:
            progress_bar -> Dict for progress bar display. Must have only tensors
            log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)

        If you didn't define a validation_step, this won't be called.
         Called at the end of the validation loop with the outputs of validation_step.

        The outputs here are strictly for the progress bar.
         If you don't need to display anything, don't return anything.
         Any keys present in 'log', 'progress_bar' or the rest of the dictionary
         are available for callbacks to access. If you want to manually set current step, you can specify it with
         'step' key in the 'log' Dict.

        Example
        -------

        With a single dataloader

        .. code-block:: python

            def validation_end(self, outputs):
                val_loss_mean = 0
                val_acc_mean = 0
                for output in outputs:
                    val_loss_mean += output['val_loss']
                    val_acc_mean += output['val_acc']

                val_loss_mean /= len(outputs)
                val_acc_mean /= len(outputs)
                tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}

                # show val_loss and val_acc in progress bar but only log val_loss
                results = {
                    'progress_bar': tqdm_dict,
                    'log': {'val_loss': val_loss_mean.item()}
                }
                return results

        With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
        one entry per dataloader, while the inner list contains the individual outputs of
        each validation step for that dataloader.

        .. code-block:: python

            def validation_end(self, outputs):
                val_loss_mean = 0
                val_acc_mean = 0
                i = 0
                for dataloader_outputs in outputs:
                    for output in dataloader_outputs:
                        val_loss_mean += output['val_loss']
                        val_acc_mean += output['val_acc']
                        i += 1

                val_loss_mean /= i
                val_acc_mean /= i
                tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}

                # show val_loss and val_acc in progress bar but only log val_loss
                results = {
                    'progress_bar': tqdm_dict,
                    'log': {'val_loss': val_loss_mean.item(), 'step': self.current_epoch}
                }
                return results

        """

    def test_end(self, outputs: list) -> dict:
        """Outputs has the appended output after each test step.

        :param outputs:  List of outputs you defined in test_step, or if there are multiple dataloaders,
         a list containing a list of outputs for each dataloader
        :return: Dict of OrderedDict with metrics to display in progress bar

        If you didn't define a test_step, this won't be called.
         Called at the end of the test step with the output of each test_step.
         The outputs here are strictly for the progress bar.
         If you don't need to display anything, don't return anything.

        Example
        -------

        .. code-block:: python

            def test_end(self, outputs):
                test_loss_mean = 0
                test_acc_mean = 0
                for output in outputs:
                    test_loss_mean += output['test_loss']
                    test_acc_mean += output['test_acc']

                test_loss_mean /= len(outputs)
                test_acc_mean /= len(outputs)
                tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}

                # show test_loss and test_acc in progress bar but only log test_loss
                results = {
                    'progress_bar': tqdm_dict,
                    'log': {'test_loss': val_loss_mean.item()}
                }
                return results

        With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
        one entry per dataloader, while the inner list contains the individual outputs of
        each validation step for that dataloader.

        .. code-block:: python

            def test_end(self, outputs):
                test_loss_mean = 0
                test_acc_mean = 0
                i = 0
                for dataloader_outputs in outputs:
                    for output in dataloader_outputs:
                        test_loss_mean += output['test_loss']
                        test_acc_mean += output['test_acc']
                        i += 1

                test_loss_mean /= i
                test_acc_mean /= i
                tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}

                # show test_loss and test_acc in progress bar but only log test_loss
                results = {
                    'progress_bar': tqdm_dict,
                    'log': {'test_loss': val_loss_mean.item()}
                }
                return results

        """

    def configure_ddp(self, model: 'LightningModule', device_ids: list) -> Module:
        r"""

        Override to init DDP in your own way or with your own wrapper.
        The only requirements are that:

        1. On a validation batch the call goes to model.validation_step.
        2. On a training batch the call goes to model.training_step.
        3. On a testing batch, the call goes to model.test_step

        Args:
            model: the LightningModule currently being optimized
            device_ids: the list of GPU ids

        Return:
            DDP wrapped model

        Example
        -------
        .. code-block:: python

            # default implementation used in Trainer
            def configure_ddp(self, model, device_ids):
                # Lightning DDP simply routes to test_step, val_step, etc...
                model = LightningDistributedDataParallel(
                    model,
                    device_ids=device_ids,
                    find_unused_parameters=True
                )
                return model


        """
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return model

    def init_ddp_connection(self, proc_rank: int, world_size: int) -> None:
        r"""

        Override to define your custom way of setting up a distributed environment.

        Lightning's implementation uses env:// init by default and sets the first node as root.

        Args:
            proc_rank: The current process rank within the node.
            world_size: Number of GPUs being use across all nodes. (num_nodes*nb_gpu_nodes).
        Example
        -------
        .. code-block:: python

            def init_ddp_connection(self):
                # use slurm job id for the port number
                # guarantees unique ports across jobs from same grid search
                try:
                    # use the last 4 numbers in the job id as the id
                    default_port = os.environ['SLURM_JOB_ID']
                    default_port = default_port[-4:]

                    # all ports should be in the 10k+ range
                    default_port = int(default_port) + 15000

                except Exception as e:
                    default_port = 12910

                # if user gave a port number, use that one instead
                try:
                    default_port = os.environ['MASTER_PORT']
                except Exception:
                    os.environ['MASTER_PORT'] = str(default_port)

                # figure out the root node addr
                try:
                    root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
                except Exception:
                    root_node = '127.0.0.2'

                root_node = self.trainer.resolve_root_node_address(root_node)
                os.environ['MASTER_ADDR'] = root_node
                dist.init_process_group(
                    'nccl',
                    rank=self.proc_rank,
                    world_size=self.world_size
                )

        """
        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        try:
            # use the last 4 numbers in the job id as the id
            default_port = os.environ['SLURM_JOB_ID']
            default_port = default_port[-4:]

            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

        except Exception:
            default_port = 12910

        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        try:
            root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
        except Exception:
            root_node = '127.0.0.2'

        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    def configure_apex(
            self,
            amp: object,
            model: 'LightningModule',
            optimizers: List[Optimizer],
            amp_level: str
    ) -> Tuple['LightningModule', List[Optimizer]]:
        r"""
        Override to init AMP your own way
        Must return a model and list of optimizers

        Args:
            amp: pointer to amp library object
            model: pointer to current lightningModule
            optimizers: list of optimizers passed in configure_optimizers()
            amp_level: AMP mode chosen ('O1', 'O2', etc...)

        Return:
            Apex wrapped model and optimizers

        Example
        -------
        .. code-block:: python

            # Default implementation used by Trainer.
            def configure_apex(self, amp, model, optimizers, amp_level):
                model, optimizers = amp.initialize(
                    model, optimizers, opt_level=amp_level,
                )

                return model, optimizers
        """
        model, optimizers = amp.initialize(
            model, optimizers, opt_level=amp_level,
        )

        return model, optimizers

    def configure_optimizers(self) -> Union[
        Optimizer, List[Optimizer], Tuple[Optimizer, ...], Tuple[List[Optimizer], list]
    ]:
        r"""
        This is where you choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or something more esoteric you might have multiple.

        If you don't define this method Lightning will automatically use Adam(lr=1e-3)

        Return: any of these 3 options:
            - Single optimizer
            - List or Tuple - List of optimizers
            - Two lists - The first list has multiple optimizers, the second a list of learning-rate schedulers

        Example
        -------

        .. code-block:: python

            # most cases (default if not defined)
            def configure_optimizers(self):
                opt = Adam(self.parameters(), lr=1e-3)
                return opt

            # multiple optimizer case (eg: GAN)
            def configure_optimizers(self):
                generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                return generator_opt, disriminator_opt

            # example with learning_rate schedulers
            def configure_optimizers(self):
                generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
                return [generator_opt, disriminator_opt], [discriminator_sched]

        .. note:: Lightning calls .backward() and .step() on each optimizer and learning rate scheduler as needed.

        .. note:: If you use 16-bit precision (use_amp=True), Lightning will automatically
            handle the optimizers for you.

        .. note:: If you use multiple optimizers, training_step will have an additional `optimizer_idx` parameter.

        .. note:: If you use LBFGS lightning handles the closure function automatically for you

        .. note:: If you use multiple optimizers, gradients will be calculated only
            for the parameters of current optimizer at each training step.

        .. note:: If you need to control how often those optimizers step or override the default .step() schedule,
            override the `optimizer_step` hook.


        """
        return Adam(self.parameters(), lr=1e-3)

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            second_order_closure: Optional[Callable] = None,
    ) -> None:
        r"""

        Override this method to adjust the default way the Trainer calls each optimizer. By default, Lightning
        calls .step() and zero_grad() as shown in the example once per optimizer.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list
            second_order_closure: closure for second order methods

        Example
        -------
        .. code-block:: python

            # DEFAULT
            def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
                optimizer.step()
                optimizer.zero_grad()

            # Alternating schedule for optimizer steps (ie: GANs)
            def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
                # update generator opt every 2 steps
                if optimizer_idx == 0:
                    if batch_idx % 2 == 0 :
                        optimizer.step()
                        optimizer.zero_grad()

                # update discriminator opt every 4 steps
                if optimizer_idx == 1:
                    if batch_idx % 4 == 0 :
                        optimizer.step()
                        optimizer.zero_grad()

                # ...
                # add as many optimizers as you want


        Here's another example showing how to use this for more advanced things such as learning-rate warm-up:

        .. code-block:: python

            # learning rate warm-up
            def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
                # warm up lr
                if self.trainer.global_step < 500:
                    lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_scale * self.hparams.learning_rate

                # update params
                optimizer.step()
                optimizer.zero_grad()

        """
        if self.trainer.use_tpu and XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        elif isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(second_order_closure)
        else:
            optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    def tbptt_split_batch(self, batch: Tensor, split_size: int) -> list:
        r"""

        When using truncated backpropagation through time, each batch must be split along the time dimension.
        Lightning handles this by default, but  for custom behavior override this function.

        Args:
            batch: Current batch
            split_size: How big the split  is

        Return:
            list of batch splits. Each split will be passed to forward_step to enable truncated
            back propagation through time. The default implementation splits root level Tensors and
            Sequences at dim=1 (i.e. time dim). It assumes that each time dim is the same length.

        Example
        -------
        .. code-block:: python

            def tbptt_split_batch(self, batch, split_size):
              splits = []
              for t in range(0, time_dims[0], split_size):
                  batch_split = []
                  for i, x in enumerate(batch):
                      if isinstance(x, torch.Tensor):
                          split_x = x[:, t:t + split_size]
                      elif isinstance(x, collections.Sequence):
                          split_x = [None] * len(x)
                          for batch_idx in range(len(x)):
                              split_x[batch_idx] = x[batch_idx][t:t + split_size]

                      batch_split.append(split_x)

                  splits.append(batch_split)

              return splits

        .. note:: Called in the training loop after on_batch_start if `truncated_bptt_steps > 0`.
            Each returned batch split is passed separately to training_step(...).

        """
        time_dims = [len(x[0]) for x in batch if isinstance(x, (torch.Tensor, collections.Sequence))]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension"
        assert all(x == time_dims[0] for x in time_dims), "Batch time dimension length is ambiguous"

        splits = []
        for t in range(0, time_dims[0], split_size):
            batch_split = []
            for i, x in enumerate(batch):
                if isinstance(x, torch.Tensor):
                    split_x = x[:, t:t + split_size]
                elif isinstance(x, collections.Sequence):
                    split_x = [None] * len(x)
                    for batch_idx in range(len(x)):
                        split_x[batch_idx] = x[batch_idx][t:t + split_size]

                batch_split.append(split_x)

            splits.append(batch_split)

        return splits

    def prepare_data(self) -> None:
        """Use this to download and prepare data.
        In distributed (GPU, TPU), this will only be called once

        This is called before requesting the dataloaders

        .. code-block:: python

            model.prepare_data()
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()

        Example
        -------

        .. code-block:: python

            def prepare_data(self):
                download_imagenet()
                clean_imagenet()
                cache_imagenet()
        """

    def train_dataloader(self) -> DataLoader:
        """Implement a PyTorch DataLoader

        :return: PyTorch DataLoader

        Return a dataloader. It will not be called every epoch unless you set
        ```Trainer(reload_dataloaders_every_epoch=True)```.

        It's recommended that all data downloads and preparation happen in prepare_data().

        .. note:: Lightning adds the correct sampler for distributed and arbitrary hardware. No need to set yourself.

        - .fit()
        - ...
        - prepare_data()
        - train_dataloader

        Example
        -------

        .. code-block:: python

            def train_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True
                )
                return loader

        """

    @data_loader
    def tng_dataloader(self):  # todo: remove in v0.8.0
        """Implement a PyTorch DataLoader.

        .. warning:: Deprecated in v0.5.0. use train_dataloader instead.
        """
        output = self.train_dataloader()
        warnings.warn("`tng_dataloader` has been renamed to `train_dataloader` since v0.5.0."
                      " and this method will be removed in v0.8.0", DeprecationWarning)
        return output

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        r"""

        Return a dataloader. It will not be called every epoch unless you set
        ```Trainer(reload_dataloaders_every_epoch=True)```.

        It's recommended that all data downloads and preparation happen in prepare_data().

        - .fit()
        - ...
        - prepare_data()
        - train_dataloader
        - val_dataloader
        - test_dataloader

        .. note:: Lightning adds the correct sampler for distributed and arbitrary hardware. No need to set yourself.

        Return:
            Single or multiple PyTorch DataLoader

        Example
        -------

        .. code-block:: python

            def test_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True
                )

                return loader

        .. note:: If you don't need a test dataset and a test_step, you don't need to implement this method.

        .. note:: If you want to change the data during every epoch DON'T use the data_loader decorator.

        """

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        r"""

        Return a dataloader. It will not be called every epoch unless you set
        ```Trainer(reload_dataloaders_every_epoch=True)```.

        It's recommended that all data downloads and preparation happen in prepare_data().

        - .fit()
        - ...
        - prepare_data()
        - train_dataloader
        - val_dataloader

        .. note:: Lightning adds the correct sampler for distributed and arbitrary hardware No need to set yourself.

        Return:
            Single or multiple PyTorch DataLoader

        Example
        -------

        .. code-block:: python

            def val_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True
                )

                return loader

            # can also return multiple dataloaders
            def val_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]

        Example
        -------

        .. code-block:: python

            @pl.data_loader
            def val_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True
                )

                return loader

            # can also return multiple dataloaders
            @pl.data_loader
            def val_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]

        .. note:: If you don't need a validation dataset and a validation_step, you don't need to implement this method.

        .. note:: If you want to change the data during every epoch DON'T use the data_loader decorator.

        .. note:: In the case where you return multiple `val_dataloaders`, the `validation_step`
            will have an argument `dataset_idx` which matches the order here.
        """

    @classmethod
    def load_from_metrics(
            cls,
            weights_path: str,
            tags_csv: str,
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None
    ) -> 'LightningModule':
        r"""
        Warning:
            Deprecated in version 0.7.0.
            You should use `load_from_checkpoint` instead.
            Will be removed in v0.9.0.
        """
        warnings.warn(
            "`load_from_metrics` method has been unified with `load_from_checkpoint` in v0.7.0."
            " The deprecated method will be removed in v0.9.0.", DeprecationWarning
        )
        return cls.load_from_checkpoint(weights_path, tags_csv=tags_csv, map_location=map_location)

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: str,
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            tags_csv: Optional[str] = None,
    ) -> 'LightningModule':
        r"""

        Primary way of loading model from a checkpoint. When Lightning saves a checkpoint
        it stores the hyperparameters in the checkpoint if you initialized your LightningModule
        with an argument called `hparams` which is a Namespace (output of using argparse
        to parse command line arguments) or dictionary of hyperparameters.

        Example
        -------
        .. code-block:: python

            from argparse import Namespace
            hparams = Namespace(**{'learning_rate': 0.1})

            model = MyModel(hparams)

            class MyModel(LightningModule):
                def __init__(self, hparams):
                    self.learning_rate = hparams.learning_rate

        Args:
            checkpoint_path: Path to checkpoint.
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in
                `torch.load <https://pytorch.org/docs/stable/torch.html#torch.load>`_.
            tags_csv: Optional path to a .csv file with two columns (key, value)
                as in this example::

                    key,value
                    drop_prob,0.2
                    batch_size,32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a .csv file with the hparams you'd like to use.
                These will be converted into a argparse.Namespace and passed into your
                LightningModule for use.

        Return:
            LightningModule with loaded weights and hyperparameters (if available).

        Example
        -------
        .. code-block:: python

            # load weights without mapping ...
            MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

            # or load weights mapping all weights from GPU 1 to GPU 0 ...
            map_location = {'cuda:1':'cuda:0'}
            MyLightningModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                map_location=map_location
            )

            # or load weights and hyperparameters from separate files.
            MyLightningModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                tags_csv='/path/to/hparams_file.csv'
            )

            # predict
            pretrained_model.eval()
            pretrained_model.freeze()
            y_hat = pretrained_model(x)
        """
        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        if tags_csv is not None:
            # add the hparams from csv file to checkpoint
            hparams = load_hparams_from_tags_csv(tags_csv)
            hparams.__setattr__('on_gpu', False)
            checkpoint['hparams'] = vars(hparams)

        model = cls._load_model_state(checkpoint)
        return model

    @classmethod
    def _load_model_state(cls, checkpoint: dict) -> 'LightningModule':
        cls_takes_hparams = 'hparams' in inspect.signature(cls.__init__).parameters
        ckpt_hparams = checkpoint.get('hparams')

        if cls_takes_hparams:
            if ckpt_hparams is not None:
                hparams = Namespace(**ckpt_hparams)
            else:
                warnings.warn(
                    f"Checkpoint does not contain hyperparameters but {cls.__name__}'s __init__ contains"
                    " argument 'hparams'. Will pass in an empty Namespace instead."
                    " Did you forget to store your model hyperparameters in self.hparams?"
                )
                hparams = Namespace()
        else:  # The user's LightningModule does not define a hparams argument
            if ckpt_hparams is None:
                hparams = None
            else:
                raise MisconfigurationException(
                    f"Checkpoint contains hyperparameters but {cls.__name__}'s __init__ is missing the"
                    " argument 'hparams'. Are you loading the correct checkpoint?"
                )

        # load the state_dict on the model automatically
        model_args = [hparams] if hparams else []
        model = cls(*model_args)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    def summarize(self, mode: str) -> None:
        model_summary = ModelSummary(self, mode=mode)
        log.info('\n' + model_summary.__str__())

    def freeze(self) -> None:
        r"""
        Freeze all params for inference

        Example
        -------
        .. code-block:: python

            model = MyLightningModule(...)
            model.freeze()

        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all params for inference.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()

        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        r"""
        Called by lightning to restore your model.
        If you saved something with **on_save_checkpoint** this is your chance to restore this.

        Args:
            checkpoint: Loaded checkpoint


        Example
        -------

        .. code-block:: python

            def on_load_checkpoint(self, checkpoint):
                # 99% of the time you don't need to implement this method
                self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']

        .. note:: Lighting auto-restores global step, epoch, and all training state including amp scaling.
            No need for you to restore anything regarding training.
        """

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        r"""

        Called by lightning when saving a  checkpoint  to give you a chance to store anything else you
        might want to  save

        Args:
            checkpoint: Checkpoint to be saved

        Example
        -------

        .. code-block:: python

            def on_save_checkpoint(self, checkpoint):
                # 99% of use cases you don't need to implement this method
                checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

        .. note:: Lighting saves all aspects of training (epoch, global step, etc...) including amp scaling. No need
            for you to store anything about training.

        """

    def get_tqdm_dict(self) -> dict:
        r"""
        Additional items to be displayed in the progress bar.

        Return:
            Dictionary with the items to be displayed in the progress bar.
        """
        tqdm_dict = {
            'loss': '{:.3f}'.format(self.trainer.avg_loss)
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict
