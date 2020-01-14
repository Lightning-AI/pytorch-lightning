import os
import warnings
import collections
import logging
from abc import ABC, abstractmethod
from argparse import Namespace

import torch
import torch.distributed as dist

from pytorch_lightning.core.decorators import data_loader
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.saving import ModelIO
from pytorch_lightning.trainer.training_io import load_hparams_from_tags_csv
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel


class LightningModule(ABC, GradInformation, ModelIO, ModelHooks):
    """
    A LightningModule has the following properties which you can access at any time

    **logger**
    A reference to the logger you passed into trainer.
    Passing a logger is optional. If you don't pass one in, Lightning will create one
     for you automatically. This logger saves logs to `/os.getcwd()/lightning_logs`::

        Trainer(logger=your_logger)


    Call it from anywhere in your LightningModule to add metrics, images, etc...
     whatever your logger supports.

    Here is an example using the TestTubeLogger (which is a wrapper
     on 'PyTorch SummaryWriter <https://pytorch.org/docs/stable/tensorboard.html>`_
     with versioned folder structure).

    .. code-block:: python

        # if logger is a tensorboard logger or TestTubeLogger
        self.logger.experiment.add_embedding(...)
        self.logger.experiment.log({'val_loss': 0.9})
        self.logger.experiment.add_scalars(...)


    **trainer**
    Last resort access to any state the trainer has.
     Changing certain properties here could affect your training run.

    .. code-block:: python

        self.trainer.optimizers
        self.trainer.current_epoch
        ...

    Debugging
    ---------

    The LightningModule also offers these tricks to help debug.

    **example_input_array**

    In the LightningModule init, you can set a dummy tensor for this property
    to get a print out of sizes coming into and out of every layer.

    .. code-block:: python

        def __init__(self):
            # put the dimensions of the first input to your system
            self.example_input_array = torch.rand(5, 28 * 28)


    """

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
        self.trainer = None
        self.logger = None
        self.example_input_array = None

        # track if gpu was requested for checkpointing
        #: True if your model is currently running on GPUs.
        #: Useful to set flags around the LightningModule for different CPU vs GPU behavior.
        self.on_gpu = False
        self.use_dp = False
        self.use_ddp = False
        self.use_ddp2 = False
        self.use_amp = False

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Expand model in into whatever you need.
        Also need to return the target
        :param x:
        :return:
        """

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """return loss, dict with metrics for tqdm

        :param batch: The output of your dataloader. A tensor, tuple or list
        :param int batch_idx: Integer displaying which batch this is
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

                # if using TestTubeLogger or TensorboardLogger you can nest scalars
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

        You can also return a -1 instead of a dict to stop the current loop. This is useful
         if you want to break out of the current training epoch early.
        """

    def training_end(self, *args, **kwargs):
        """return loss, dict with metrics for tqdm

        :param outputs: What you return in `training_step`.
        :return dict: dictionary with loss key and optional log, progress keys:
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
        pass

    def validation_step(self, *args, **kwargs):
        """return whatever outputs will need to be aggregated in validation_end

        :param batch: The output of your dataloader. A tensor, tuple or list
        :param int batch_idx: Integer displaying which batch this is
        :param int dataloader_idx: Integer displaying which dataloader this is (only if multiple val datasets used)
        :return dict: Dict or OrderedDict - passed to the validation_end step

        .. code-block:: python

            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx)

            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idxdx)

        If you don't need to validate you don't need to implement this method.
         In this step you'd normally generate examples or calculate anything of interest such as accuracy.

        When the validation_step is called, the model has been put in eval mode and PyTorch gradients
         have been disabled. At the end of validation, model goes back to training mode and gradients are enabled.

        The dict you return here will be available in the `validation_end` method.

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

        The `dataset_idx` corresponds to the order of datasets returned in `val_dataloader`.
        """
        pass

    def test_step(self, *args, **kwargs):
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
        If you don't need to test you don't need to implement this method. In this step you'd normally
         generate examples or calculate anything of interest such as accuracy.

        When the validation_step is called, the model has been put in eval mode and PyTorch gradients
         have been disabled. At the end of validation, model goes back to training mode and gradients are enabled.

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
        pass

    def validation_end(self, outputs):
        """Outputs has the appended output after each validation step.

        :param outputs: List of outputs you defined in validation_step, or if there are multiple dataloaders,
         a list containing a list of outputs for each dataloader
        :return dict: Dictionary or OrderedDict with optional:
            progress_bar -> Dict for progress bar display. Must have only tensors
            log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)

        If you didn't define a validation_step, this won't be called.
         Called at the end of the validation loop with the outputs of validation_step.

        The outputs here are strictly for the progress bar.
         If you don't need to display anything, don't return anything.
         Any keys present in 'log', 'progress_bar' or the rest of the dictionary
         are available for callbacks to access.

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
                    'log': {'val_loss': val_loss_mean.item()}
                }
                return results

        """
        pass

    def test_end(self, outputs):
        """Outputs has the appended output after each test step.

        :param outputs:  List of outputs you defined in test_step, or if there are multiple dataloaders,
         a list containing a list of outputs for each dataloader
        :return dict: Dict of OrderedDict with metrics to display in progress bar

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
        pass

    def configure_ddp(self, model, device_ids):
        """Override to init DDP in a different way or use your own wrapper.

        :param model:
        :param device_ids:
        :return: DDP wrapped model

        Overwrite to define your own DDP implementation init.
        The only requirement is that:
        1. On a validation batch the call goes to model.validation_step.
        2. On a training batch the call goes to model.training_step.
        3. On a testing batch, the call goes to model.test_step

        .. code-block:: python

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

    def init_ddp_connection(self, proc_rank, world_size):
        """Connect all procs in the world using the env:// init
        Use the first node as the root address

        Override to init DDP in your own way.

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
                dist.init_process_group('nccl', rank=self.proc_rank, world_size=self.world_size)

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

    def configure_apex(self, amp, model, optimizers, amp_level):
        """
        Override to init AMP your own way
        Must return a model and list of optimizers
        :param amp:
        :param model:
        :param optimizers:
        :param amp_level:
        :return: Apex wrapped model and optimizers

        Overwrite to define your own Apex implementation init.

        .. code-block:: python

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

    @abstractmethod
    def configure_optimizers(self):
        """Return a list of optimizers and a list of schedulers (could be empty)

        :return: any of these 3 options:
            - Single optimizer
            - List or Tuple - List of optimizers
            - Two lists - The first list has multiple optimizers, the second a list of learning-rate schedulers

        Set up as many optimizers and (optionally) learning rate schedulers as you need.
         Normally you'd need one. But in the case of GANs or something more esoteric you might have multiple.
         Lightning will call .backward() and .step() on each one in every epoch.
         If you use 16 bit precision it will also handle that.

        .. note:: If you use multiple optimizers, training_step will have an additional `optimizer_idx` parameter.

        .. note:: If you use LBFGS lightning handles the closure function automatically for you.
        
        .. note:: If you use multiple optimizers, gradients will be calculated only for the parameters of current optimizer at each training step.

        Example
        -------

        .. code-block:: python

            # most cases
            def configure_optimizers(self):
                opt = Adam(self.parameters(), lr=0.01)
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

        If you need to control how often those optimizers step or override the default .step() schedule,
         override the `optimizer_step` hook.

        """

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        """Do something instead of the standard optimizer behavior

        :param int epoch:
        :param int batch_idx:
        :param optimizer:
        :param optimizer_idx:
        :param second_order_closure: closure for second order methods
        :return:

        Calls `.step()` and `.zero_grad` for each optimizer.
        You can override this method to adjust how you do the optimizer step for each optimizer

        Called once per optimizer

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


        This step allows you to do a lot of non-standard training tricks such as learning-rate warm-up:

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
        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(second_order_closure)
        else:
            optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    def tbptt_split_batch(self, batch, split_size):
        """
        Return list of batch splits. Each split will be passed to forward_step to enable truncated
        back propagation through time. The default implementation splits root level Tensors and
        Sequences at dim=1 (i.e. time dim). It assumes that each time dim is the same length.

        :param batch:
        :param split_size:
        :return:

        Called in the training loop after on_batch_start if `truncated_bptt_steps > 0`.
         Each returned batch split is passed separately to training_step(...).

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
        """
        time_dims = [len(x[0]) for x in batch if isinstance(
            x, torch.Tensor) or isinstance(x, collections.Sequence)]
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

    @data_loader
    @abstractmethod
    def train_dataloader(self):
        """Implement a PyTorch DataLoader

        :return: PyTorch DataLoader

        Called by lightning during training loop. Make sure to use the @pl.data_loader decorator,
         this ensures not calling this function until the data are needed.
         If you want to change the data during every epoch DON'T use the data_loader decorator.

        Example
        -------

        .. code-block:: python

            @pl.data_loader
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
    def tng_dataloader(self):
        """Implement a PyTorch DataLoader.

        .. warning:: Deprecated in v0.5.0. use train_dataloader instead.
        """
        output = self.train_dataloader()
        warnings.warn("`tng_dataloader` has been renamed to `train_dataloader` since v0.5.0"
                      " and will be removed in v0.8.0", DeprecationWarning)
        return output

    @data_loader
    def test_dataloader(self):
        """Implement a PyTorch DataLoader.

        :return: PyTorch DataLoader

        If you don't need a test dataset and a test_step, you don't need to implement this method.

        Called by lightning during test loop. Make sure to use the @pl.data_loader decorator,
         this ensures not calling this function until the data are needed.
         If you want to change the data during every epoch DON'T use the data_loader decorator.

        Example
        -------

        .. code-block:: python

            @pl.data_loader
            def test_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True
                )

                return loader

        """
        return None

    @data_loader
    def val_dataloader(self):
        """Implement a PyTorch DataLoader.

        :return: PyTorch DataLoader or list of PyTorch Dataloaders.

        If you don't need a validation dataset and a validation_step, you don't need to implement this method.

        Called by lightning during validation loop. Make sure to use the @pl.data_loader decorator,
         this ensures not calling this function until the data are needed.
         If you want to change the data during every epoch DON'T use the data_loader decorator.

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

        In the case where you return multiple `val_dataloaders`, the `validation_step`
         will have an arguement `dataset_idx` which matches the order here.
        """
        return None

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv, map_location=None):
        """Primary way of loading model from csv weights path.

        :param str weights_path: Path to a PyTorch checkpoint
        :param str tags_csv: Path to meta_tags.csv file generated by the test-tube Experiment
        :param dict map_location: A dictionary mapping saved weight GPU devices to new GPU devices
            for mapping storage {'cuda:1':'cuda:0'}
        :return: The pretrained LightningModule

        If you're using test tube, there is an alternate method which uses the meta_tags.csv
        file from test-tube to rebuild the model. The meta_tags.csv file can be found in the
        test-tube experiment save_dir.

        .. code-block:: python

            pretrained_model = MyLightningModule.load_from_metrics(
                weights_path='/path/to/pytorch_checkpoint.ckpt',
                tags_csv='/path/to/test_tube/experiment/version/meta_tags.csv',
                on_gpu=True,
                map_location=None
            )

            # predict
            pretrained_model.eval()
            pretrained_model.freeze()
            y_hat = pretrained_model(x)

        This is the easiest/fastest way which loads hyperparameters and weights from a checkpoint,
        such as the one saved by the `ModelCheckpoint` callback

        .. code-block:: python

            pretrained_model = MyLightningModule.load_from_checkpoint(
                checkpoint_path='/path/to/pytorch_checkpoint.ckpt'
            )

            # predict
            pretrained_model.eval()
            pretrained_model.freeze()
            y_hat = pretrained_model(x)

        """
        hparams = load_hparams_from_tags_csv(tags_csv)
        hparams.__setattr__('on_gpu', False)

        if map_location is not None:
            checkpoint = torch.load(weights_path, map_location=map_location)
        else:
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        # load the state_dict on the model automatically
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        """
        Primary way of loading model from a checkpoint
        :param checkpoint_path:
        :param map_location: dic for mapping storage {'cuda:1':'cuda:0'}
        :return:
        """

        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        try:
            ckpt_hparams = checkpoint['hparams']
        except KeyError:
            raise IOError(
                "Checkpoint does not contain hyperparameters. Are your model hyperparameters stored"
                "in self.hparams?"
            )
        hparams = Namespace(**ckpt_hparams)

        # load the state_dict on the model automatically
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    def summarize(self, mode):
        model_summary = ModelSummary(self, mode=mode)
        logging.info('\n' + model_summary.__str__())

    def freeze(self):
        """Freeze all params for inference

        .. code-block:: python

            model = MyLightningModule(...)
            model.freeze()

        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self):
        """Unfreeze all params for inference.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()

        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def on_load_checkpoint(self, checkpoint):
        """

        :param checkpoint:

        Called by lightning to restore your model. Lighting auto-restores global step, epoch, etc...
         It also restores the model state_dict.
         If you saved something with **on_save_checkpoint** this is your chance to restore this.

        Example
        -------

        .. code-block:: python

            def on_load_checkpoint(self, checkpoint):
                # 99% of the time you don't need to implement this method
                self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']

        """
        pass

    def on_save_checkpoint(self, checkpoint):
        """

        :param checkpoint:

        Called by lightning to checkpoint your model. Lightning saves the training state
         (current epoch, global_step, etc) and also saves the model state_dict.
         If you want to save anything else, use this method to add your own key-value pair.

        Example
        -------

        .. code-block:: python

            def on_save_checkpoint(self, checkpoint):
                # 99% of use cases you don't need to implement this method
                checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

        """
        pass
