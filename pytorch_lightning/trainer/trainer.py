"""
The trainer handles all the logic for running a val loop, training loop, distributing, etc.. .
"""

import os
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.optim.optimizer import Optimizer

from pytorch_lightning.trainer.amp_mixin import TrainerAMPMixin
from pytorch_lightning.trainer.callback_config_mixin import TrainerCallbackConfigMixin
from pytorch_lightning.trainer.data_loading_mixin import TrainerDataLoadingMixin
from pytorch_lightning.trainer.ddp_mixin import TrainerDDPMixin
from pytorch_lightning.trainer.dp_mixin import TrainerDPMixin
from pytorch_lightning.trainer.dp_mixin import (
    parse_gpu_ids,
    determine_root_gpu_device
)
from pytorch_lightning.trainer.evaluation_loop_mixin import TrainerEvaluationLoopMixin
from pytorch_lightning.trainer.logging_mixin import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks_mixin import TrainerModelHooksMixin
from pytorch_lightning.trainer.train_loop_mixin import TrainerTrainLoopMixin
from pytorch_lightning.trainer.trainer_io import TrainerIOMixin
<<<<<<< HEAD
from pytorch_lightning.pt_overrides.override_data_parallel import (
    LightningDistributedDataParallel, LightningDataParallel)
from pytorch_lightning.callbacks import GradientAccumulationScheduler, \
    ReduceLROnPlateauScheduler, ModelCheckpoint, EarlyStopping
=======
from pytorch_lightning.trainer.training_tricks_mixin import TrainerTrainingTricksMixin
>>>>>>> 28c3bcb0c018d9fadf82085939c8392bf2e1fa86
from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class Trainer(TrainerIOMixin,
              TrainerDDPMixin,
              TrainerDPMixin,
              TrainerDataLoadingMixin,
              TrainerAMPMixin,
              TrainerEvaluationLoopMixin,
              TrainerTrainLoopMixin,
              TrainerLoggingMixin,
              TrainerTrainingTricksMixin,
              TrainerCallbackConfigMixin,
              TrainerModelHooksMixin):

    def __init__(self,
                 logger=True,
                 checkpoint_callback=True,
                 early_stop_callback=True,
                 default_save_path=None,
                 gradient_clip_val=0,
                 gradient_clip=None,  # backward compatible
                 process_position=0,
                 nb_gpu_nodes=1,
                 gpus=None,
                 log_gpu_memory=None,
                 show_progress_bar=True,
                 overfit_pct=0.0,
                 track_grad_norm=-1,
                 check_val_every_n_epoch=1,
                 fast_dev_run=False,
                 accumulate_grad_batches=1,
                 max_nb_epochs=1000,
                 min_nb_epochs=1,
                 train_percent_check=1.0,
                 val_percent_check=1.0,
                 test_percent_check=1.0,
                 val_check_interval=1.0,
                 log_save_interval=100,
                 row_log_interval=10,
                 add_row_log_interval=None,  # backward compatible
                 distributed_backend=None,
                 use_amp=False,
                 print_nan_grads=False,
                 weights_summary='full',
                 weights_save_path=None,
                 amp_level='O1',
                 nb_sanity_val_steps=5):
        """

        :param logger: Logger for experiment tracking
        :param checkpoint_callback: Callback for checkpointing
        :param early_stop_callback: Callback for early stopping
        :param default_save_path: Default path for logs+weights if no logger/ckpt_callback passed
        :param gradient_clip_val: int. 0 means don't clip.
        :param gradient_clip: int. 0 means don't clip. Deprecated.
        :param process_position: shown in the tqdm bar
        :param nb_gpu_nodes: number of GPU nodes
        :param gpus: int. (ie: 2 gpus) OR list to specify which GPUs [0, 1] OR '0,1'
            OR '-1' / -1 to use all available gpus
        :param log_gpu_memory: str. None, 'min_max', 'all'
        :param show_progress_bar: Bool. If true shows tqdm bar
        :param overfit_pct: float. uses this much of all datasets
        :param track_grad_norm: int. -1 no tracking. Otherwise tracks that norm
        :param check_val_every_n_epoch: int. check val every n train epochs
        :param fast_dev_run: Bool. runs full iteration over everything to find bugs
        :param accumulate_grad_batches: int. Accumulates grads every k batches
        :param max_nb_epochs: int.
        :param min_nb_epochs: int.
        :param train_percent_check: int. How much of train set to check
        :param val_percent_check: int. How much of val set to check
        :param test_percent_check: int. How much of test set to check
        :param val_check_interval: float/int. If float, % of tng epoch. If int, check every n batch
        :param log_save_interval: int. Writes logs to disk this often
        :param row_log_interval: int. How often to add logging rows
        :param add_row_log_interval: int. How often to add logging rows. Deprecated.
        :param distributed_backend: str. Options: 'dp', 'ddp', 'ddp2'.
        :param use_amp: Bool. If true uses apex for 16bit precision
        :param print_nan_grads: Bool. Prints nan gradients
        :param weights_summary: str. Options: 'full', 'top', None to not print.
        :param weights_save_path: Bool. Where to save weights if on cluster
        :param amp_level: str. Check nvidia docs for level
        :param nb_sanity_val_steps: int. How many val steps before a full train loop.
        """
        # Transfer params
        self.nb_gpu_nodes = nb_gpu_nodes
        self.log_gpu_memory = log_gpu_memory
        if not (gradient_clip is None):
            # Backward compatibility
            warnings.warn("gradient_clip has renamed to gradient_clip_val since v0.5.0",
                          DeprecationWarning)
            gradient_clip_val = gradient_clip
        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.track_grad_norm = track_grad_norm
        self.on_gpu = gpus is not None and torch.cuda.is_available()
        self.process_position = process_position
        self.weights_summary = weights_summary
        self.max_nb_epochs = max_nb_epochs
        self.min_nb_epochs = min_nb_epochs
        self.nb_sanity_val_steps = nb_sanity_val_steps
        self.print_nan_grads = print_nan_grads

        self.fast_dev_run = fast_dev_run
        if self.fast_dev_run:
            self.nb_sanity_val_steps = 1
            self.max_nb_epochs = 1
            m = '''
            Running in fast_dev_run mode: will run a full train,
            val loop using a single batch
            '''
            print(m)

        # set default save path if user didn't provide one
        self.default_save_path = default_save_path
        if self.default_save_path is None:
            self.default_save_path = os.getcwd()

        # training bookeeping
        self.total_batch_nb = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_nb = 0
        self.tqdm_metrics = {}
        self.callback_metrics = {}
        self.nb_val_batches = 0
        self.nb_training_batches = 0
        self.nb_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None
        self.is_iterable_train_dataloader = False

        # training state
        self.model = None
        self.testing = False
        self.lr_schedulers = []
        self.optimizers = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure early stop callback
        # creates a default one if none passed in
        self.early_stop_callback = None
<<<<<<< HEAD
        if early_stop_callback is True:
            self.early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=True,
                mode='min'
            )
            self.enable_early_stop = True
        elif not early_stop_callback:
            self.early_stop_callback = None
            self.enable_early_stop = False
        else:
            self.early_stop_callback = early_stop_callback
            self.enable_early_stop = True
        self.val_loss_drop_lr_callback = None

        # configure logger
        if logger is True:
            # default logger
            self.logger = TestTubeLogger(
                save_dir=self.default_save_path,
                version=self.slurm_job_id,
                name='lightning_logs'
            )
            self.logger.rank = 0
        elif logger is False:
            self.logger = None
        else:
            self.logger = logger
            self.logger.rank = 0
=======
        self.configure_early_stopping(early_stop_callback, logger)
>>>>>>> 28c3bcb0c018d9fadf82085939c8392bf2e1fa86

        # configure checkpoint callback
        self.checkpoint_callback = checkpoint_callback
        self.weights_save_path = weights_save_path

        # accumulated grads
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # allow int, string and gpu list
        self.data_parallel_device_ids = parse_gpu_ids(gpus)
        self.root_gpu = determine_root_gpu_device(self.data_parallel_device_ids)

        # distributed backend choice
        self.use_ddp = False
        self.use_ddp2 = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = distributed_backend
        self.set_distributed_mode(distributed_backend, nb_gpu_nodes)

        # init flags for SLURM+ddp to work
        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0
        self.configure_slurm_ddp(nb_gpu_nodes)

        # nvidia setup
        self.set_nvidia_flags(self.is_slurm_managing_tasks, self.data_parallel_device_ids)

        # can't init progress bar here because starting a new process
        # means the progress_bar won't survive pickling
        self.show_progress_bar = show_progress_bar

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        if not (add_row_log_interval is None):
            # backward compatibility
            warnings.warn("gradient_clip has renamed to gradient_clip_val since v0.5.0",
                          DeprecationWarning)
            row_log_interval = add_row_log_interval
        self.row_log_interval = row_log_interval

        # how much of the data to use
        self.determine_data_use_amount(train_percent_check, val_percent_check,
                                       test_percent_check, overfit_pct)

        # 16 bit mixed precision training using apex
        self.amp_level = amp_level
        self.init_amp(use_amp)

    @property
    def slurm_job_id(self):
        try:
            job_id = os.environ['SLURM_JOB_ID']
            job_id = int(job_id)
        except Exception as e:
            job_id = None
        return job_id

    def __parse_gpu_ids(self, gpus):
        """
        :param gpus: Int, string or list of ids
        :return:
        """
        # if gpus = -1 then use all available devices
        # otherwise, split the string using commas
        if gpus is not None:
            if type(gpus) is list:
                gpus = gpus
            elif type(gpus) is str:
                if gpus == '-1':
                    gpus = list(range(0, torch.cuda.device_count()))
                else:
                    gpus = [int(x.strip()) for x in gpus.split(',')]
            elif type(gpus) is int:
                gpus = gpus
            else:
                raise Exception('gpus has to be a string, int or list of ints')

        return gpus

    def __set_root_gpu(self, gpus):
        if gpus is None:
            return None

        # set root gpu
        root_gpu = 0
        if type(gpus) is list:
            root_gpu = gpus[0]

        return root_gpu

    @property
    def num_gpus(self):
        gpus = self.data_parallel_device_ids
        if gpus is None:
            return 0
        else:
            return len(gpus)

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp or self.use_ddp2

    @property
    def training_tqdm_dict(self):
        """
        Read-only for tqdm metrics
        :return:
        """
        tqdm_dict = {
            'loss': '{0:.3f}'.format(self.avg_loss),
            'epoch': '{}'.format(self.current_epoch),
            'batch_nb': '{}'.format(self.batch_nb),
        }

        if self.logger is not None and self.logger.version is not None:
            tqdm_dict['v_nb'] = self.logger.version

        tqdm_dict.update(self.tqdm_metrics)

        if self.on_gpu:
            tqdm_dict['gpu'] = '{}'.format(torch.cuda.current_device())

        return tqdm_dict

    @property
    def tng_tqdm_dic(self):
        """
        * Deprecated in v0.5.0. use training_tqdm_dict instead. *
        :return:
        """
        warnings.warn("tng_tqdm_dict has renamed to training_tqdm_dict since v0.5.0",
                      DeprecationWarning)
        return self.training_tqdm_dict

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(self, model):
        # when using multi-node or DDP within a node start each module in a separate process
        if self.use_ddp2:
            task = int(os.environ['SLURM_LOCALID'])
            self.ddp_train(task, model)

        elif self.use_ddp:
            if self.is_slurm_managing_tasks:
                task = int(os.environ['SLURM_LOCALID'])
                self.ddp_train(task, model)
            else:
                mp.spawn(self.ddp_train, nprocs=self.num_gpus, args=(model,))

        # 1 gpu or dp option triggers training using DP module
        # easier to avoid NCCL issues
        elif self.use_dp:
            self.dp_train(model)

        elif self.single_gpu:
            self.single_gpu_train(model)

        # ON CPU
        else:
            # run through amp wrapper
            if self.use_amp:
                raise MisconfigurationException('amp + cpu is not supported.'
                                                ' Please use a GPU option')

            # CHOOSE OPTIMIZER
            # allow for lr schedulers as well
            self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

            self.run_pretrain_routine(model)

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return 1

    def init_optimizers(self, optimizers):

        # single optimizer
        if isinstance(optimizers, Optimizer):
            return [optimizers], []

        # two lists
        elif len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            lr_schedulers, self.val_loss_drop_lr_callback = self.configure_schedulers(lr_schedulers)
            return optimizers, lr_schedulers

        # single list or tuple
        elif isinstance(optimizers, list) or isinstance(optimizers, tuple):
            return optimizers, []

<<<<<<< HEAD
    def configure_schedulers(self, schedulers):
        for i in range(len(schedulers)):
            if isinstance(schedulers[i], torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss_drop_lr_callback = ReduceLROnPlateauScheduler(schedulers.pop(i),
                                                                       monitor='val_loss')
                return schedulers, val_loss_drop_lr_callback
        return schedulers, None

    def __single_gpu_train(self, model):
        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        model.cuda(self.root_gpu)

        if self.use_amp:
            # An example
            model, optimizers = amp.initialize(
                model, self.optimizers, opt_level=self.amp_level,
            )
            self.optimizers = optimizers

        self.__run_pretrain_routine(model)

    def __dp_train(self, model):

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        model.cuda(self.root_gpu)

        # check for this bug (amp + dp + !01 doesn't work)
        # https://github.com/NVIDIA/apex/issues/227
        if self.use_dp and self.use_amp:
            m = f"""
            Amp level {self.amp_level} with DataParallel is not supported.
            See this note from NVIDIA for more info: https://github.com/NVIDIA/apex/issues/227.
            We recommend you switch to ddp if you want to use amp
            """
            raise MisconfigurationException(m)

        # create list of device ids
        device_ids = self.data_parallel_device_ids
        if type(device_ids) is int:
            device_ids = list(range(device_ids))

        model = LightningDataParallel(model, device_ids=device_ids)

        self.__run_pretrain_routine(model)

    def ddp_train(self, gpu_nb, model):
        """
        Entry point into a DP thread
        :param gpu_nb:
        :param model:
        :param cluster_obj:
        :return:
        """
        # node rank using relative slurm id
        # otherwise default to node rank 0
        try:
            node_id = os.environ['SLURM_NODEID']
            self.node_rank = int(node_id)
        except Exception:
            self.node_rank = 0

        # show progressbar only on progress_rank 0
        self.show_progress_bar = self.show_progress_bar and self.node_rank == 0 and gpu_nb == 0

        # determine which process we are and world size
        if self.use_ddp:
            self.proc_rank = self.node_rank * self.num_gpus + gpu_nb
            self.world_size = self.nb_gpu_nodes * self.num_gpus

        elif self.use_ddp2:
            self.proc_rank = self.node_rank
            self.world_size = self.nb_gpu_nodes

        # let the exp know the rank to avoid overwriting logs
        if self.logger is not None:
            self.logger.rank = self.proc_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.__init_tcp_connection()

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        # MODEL
        # copy model to each gpu
        if self.distributed_backend == 'ddp':
            torch.cuda.set_device(gpu_nb)
        model.cuda(gpu_nb)

        # set model properties before going into wrapper
        model.trainer = self
        model.on_gpu = self.on_gpu
        model.use_dp = self.use_dp
        model.use_ddp2 = self.use_ddp2
        model.use_ddp = self.use_ddp
        model.use_amp = self.use_amp
        model.testing = self.testing

        # override root GPU
        self.root_gpu = gpu_nb

        # AMP
        # run through amp wrapper before going to distributed DP
        if self.use_amp:
            # An example
            model, optimizers = amp.initialize(
                model, self.optimizers, opt_level=self.amp_level,
            )
            self.optimizers = optimizers

        # DDP2 uses all GPUs on the machine
        if self.distributed_backend == 'ddp':
            device_ids = [gpu_nb]
        elif self.use_ddp2:
            device_ids = None

        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )

        # continue training routine
        self.__run_pretrain_routine(model)

    def __init_tcp_connection(self):
        """
        Connect all procs in the world using the env:// init
        Use the first node as the root address
        :param port:
        :param tries:
        :return:
        """

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

        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group("nccl", rank=self.proc_rank, world_size=self.world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node

    def __run_pretrain_routine(self, model):
=======
    def run_pretrain_routine(self, model):
>>>>>>> 28c3bcb0c018d9fadf82085939c8392bf2e1fa86
        """
        Sanity check a few things before starting actual training
        :param model:
        :return:
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.copy_trainer_model_properties(ref_model)

        # link up experiment object
        if self.logger is not None:
            ref_model.logger = self.logger

            # save exp to get started
            if hasattr(ref_model, "hparams"):
                self.logger.log_hyperparams(ref_model.hparams)

            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            dist.barrier()

        # set up checkpoint callback
        self.configure_checkpoint_callback()

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # init training constants
        self.layout_bookeeping()

        # print model summary
        if self.proc_rank == 0 and self.weights_summary is not None:
            if self.weights_summary in ['full', 'top']:
                ref_model.summarize(mode=self.weights_summary)
            else:
                m = "weights_summary can be None, 'full' or 'top'"
                raise MisconfigurationException(m)

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_weights(model)

        # progress bar init
        if self.show_progress_bar:
            self.progress_bar = tqdm.tqdm(0, position=self.process_position)

        # when testing requested only run test and return
        if self.testing:
            self.run_evaluation(test=True)
            return

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        ref_model.on_sanity_check_start()
        if self.get_val_dataloaders() is not None and self.nb_sanity_val_steps > 0:
            # reset progress_bar limit for sanity check
            if self.show_progress_bar:
                self.progress_bar.reset(self.nb_sanity_val_steps)

            self.evaluate(model, self.get_val_dataloaders(), self.nb_sanity_val_steps, self.testing)

<<<<<<< HEAD
        # ---------------------------
        # CORE TRAINING LOOP
        # ---------------------------
        self.__train()

    def __train(self):
        # run all epochs
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(self.get_train_dataloader().sampler, 'set_epoch'):
                self.get_train_dataloader().sampler.set_epoch(epoch_nb)

            # get model
            model = self.__get_model()

            # update training progress in trainer and model
            model.current_epoch = epoch_nb
            self.current_epoch = epoch_nb
            self.total_batches = self.nb_training_batches + self.nb_val_batches
            self.batch_loss_value = 0  # accumulated grads

            # limit the number of batches to 1 in fast_dev_run
            if self.fast_dev_run:
                self.total_batches = 1

            # init progress_bar when requested
            if self.show_progress_bar:
                self.progress_bar.reset(self.total_batches)

            # changing gradient according accumulation_scheduler
            self.accumulation_scheduler.on_epoch_begin(epoch_nb, self)

            # -----------------
            # RUN TNG EPOCH
            # -----------------
            self.run_training_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(epoch=self.current_epoch)

            # early stopping
            met_min_epochs = epoch_nb > self.min_nb_epochs
            if self.enable_early_stop and (met_min_epochs or self.fast_dev_run):
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch_nb,
                                                                    logs=self.callback_metrics)
                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    return

        if self.logger is not None:
            self.logger.finalize("success")

    def run_training_epoch(self):
        # before epoch hook
        if self.__is_function_implemented('on_epoch_start'):
            model = self.__get_model()
            model.on_epoch_start()

        # run epoch
        for batch_nb, batch in enumerate(self.get_train_dataloader()):
            self.batch_nb = batch_nb
            self.global_step += 1

            model = self.__get_model()
            model.global_step = self.global_step

            # stop when the flag is changed or we've gone past the amount
            #  requested in the batches
            self.total_batch_nb += 1
            met_batch_limit = batch_nb > self.nb_training_batches
            if met_batch_limit:
                break

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            output = self.__run_training_batch(batch, batch_nb)
            batch_result, grad_norm_dic, batch_step_metrics = output
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            is_val_check_batch = (batch_nb + 1) % self.val_check_batch == 0
            can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
            should_check_val = ((is_val_check_batch or early_stop_epoch) and can_check_epoch)

            # fast_dev_run always forces val checking after train batch
            if self.fast_dev_run or should_check_val:
                self.__run_evaluation(test=self.testing)

            # when logs should be saved
            should_save_log = (batch_nb + 1) % self.log_save_interval == 0 or early_stop_epoch
            if should_save_log or self.fast_dev_run:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()

            # when metrics should be logged
            should_log_metrics = batch_nb % self.row_log_interval == 0 or early_stop_epoch
            if should_log_metrics or self.fast_dev_run:

                # logs user requested information to logger
                self.__log_metrics(batch_step_metrics, grad_norm_dic)

            # end epoch early
            if early_stop_epoch or self.fast_dev_run:
                break

        # epoch end hook
        if self.__is_function_implemented('on_epoch_end'):
            model = self.__get_model()
            model.on_epoch_end()

    def __log_metrics(self, metrics, grad_norm_dic):
        """
        Logs the metric dict passed in
        :param metrics:
        :param grad_norm_dic:
        :return:
        """
        # added metrics by Lightning for convenience
        metrics['epoch'] = self.current_epoch

        # add gpu memory
        if self.on_gpu and self.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.log_gpu_memory)
            metrics.update(mem_map)

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.__metrics_to_scalars(metrics)
=======
        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()
>>>>>>> 28c3bcb0c018d9fadf82085939c8392bf2e1fa86

        # CORE TRAINING LOOP
        self.train()

    def test(self, model=None):
        self.testing = True
        if model is not None:
            self.fit(model)
        else:
<<<<<<< HEAD
            self.__run_evaluation(test=True)

    def __metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.__metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def __log_vals_blacklist(self):
        """avoid logging some vals lightning uses to maintain state"""
        blacklist = {'batch_nb', 'v_nb', 'gpu'}
        return blacklist

    def transfer_batch_to_gpu(self, batch, gpu_id):
        # base case: object can be directly moved using `cuda` or `to`
        if callable(getattr(batch, 'cuda', None)):
            return batch.cuda(gpu_id)

        elif callable(getattr(batch, 'to', None)):
            return batch.to(torch.device('cuda', gpu_id))

        # when list
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch

        # when tuple
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)

        # when dict
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)

            return batch

        # nothing matches, return the value as is without transform
        return batch

    def __training_forward(self, batch, batch_nb, opt_idx):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_nb:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_nb]
        if len(self.optimizers) > 1:
            args.append(opt_idx)

        if self.use_ddp or self.use_ddp2:
            output = self.model(*args)
        elif self.use_dp:
            output = self.model(*args)
        elif self.single_gpu:
            gpu_id = 0
            if type(self.data_parallel_device_ids) is list:
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        else:
            output = self.model.training_step(*args)

        # format and reduce outputs accordingly
        output = self.__process_output(output, train=True)
        loss, progress_bar_metrics, log_metrics, callback_metrics = output
        return loss, progress_bar_metrics, log_metrics, callback_metrics

    def __process_output(self, output, train=False):
        """
        Reduces output according to the training mode.
        Separates loss from logging and tqdm metrics
        :param output:
        :return:
        """
        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        for k, v in output.items():
            if k not in ['progress_bar', 'log']:
                callback_metrics[k] = v

        if train and self.use_dp or self.use_ddp2:
            nb_gpus = self.num_gpus
            callback_metrics = reduce_distributed_output(callback_metrics, nb_gpus)

        for k, v in callback_metrics.items():
            callback_metrics[k] = v.item()

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for tqdm when using dp
            if train and self.use_dp or self.use_ddp2:
                nb_gpus = self.num_gpus
                progress_output = reduce_distributed_output(progress_output, nb_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for tqdm when using dp
            if train and(self.use_dp or self.use_ddp2):
                nb_gpus = self.num_gpus
                log_output = reduce_distributed_output(log_output, nb_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if type(output) is torch.Tensor:
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp or self.use_ddp2:
                loss = reduce_distributed_output(loss, self.num_gpus)

        return loss, progress_bar_metrics, log_metrics, callback_metrics

    def __clip_gradients(self):
        if self.gradient_clip_val > 0:
            model = self.__get_model()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

    def __print_nan_grads(self):
        model = self.__get_model()
        for param in model.parameters():
            if torch.isnan(param.grad.float()).any():
                print(param, param.grad)

    def __run_training_batch(self, batch, batch_nb):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        all_callback_metrics = []

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic

        # hook
        if self.__is_function_implemented('on_batch_start'):
            model_ref = self.__get_model()
            response = model_ref.on_batch_start(batch)

            if response == -1:
                return -1, grad_norm_dic

        if self.show_progress_bar:
            self.progress_bar.update(1)

        # call training_step once per optimizer
        for opt_idx, optimizer in enumerate(self.optimizers):

            # wrap the forward step in a closure so second order methods work
            def optimizer_closure():
                # forward pass
                output = self.__training_forward(batch, batch_nb, opt_idx)
                closure_loss, progress_bar_metrics, log_metrics, callback_metrics = output

                # track metrics for callbacks
                all_callback_metrics.append(callback_metrics)

                # track progress bar metrics
                self.__add_tqdm_metrics(progress_bar_metrics)
                all_log_metrics.append(log_metrics)

                # accumulate loss
                # (if accumulate_grad_batches = 1 no effect)
                closure_loss = closure_loss / self.accumulate_grad_batches

                # backward pass
                if self.use_amp:
                    with amp.scale_loss(closure_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    closure_loss.backward()

                # insert after step hook
                if self.__is_function_implemented('on_after_backward'):
                    model_ref = self.__get_model()
                    model_ref.on_after_backward()

                return closure_loss

            # calculate loss
            loss = optimizer_closure()

            # nan grads
            if self.print_nan_grads:
                self.__print_nan_grads()

            # track total loss for logging (avoid mem leaks)
            self.batch_loss_value += loss.item()

            # gradient update with accumulated gradients
            if (self.batch_nb + 1) % self.accumulate_grad_batches == 0:

                # track gradient norms when requested
                if batch_nb % self.row_log_interval == 0:
                    if self.track_grad_norm > 0:
                        model = self.__get_model()
                        grad_norm_dic = model.grad_norm(self.track_grad_norm)

                # clip gradients
                self.__clip_gradients()

                # calls .step(), .zero_grad()
                # override function to modify this behavior
                model = self.__get_model()
                model.optimizer_step(self.current_epoch, batch_nb,
                                     optimizer, opt_idx, optimizer_closure)

                # calculate running loss for display
                self.running_loss.append(self.batch_loss_value)
                self.batch_loss_value = 0
                self.avg_loss = np.mean(self.running_loss[-100:])

                # update progress bar
                if self.show_progress_bar:
                    # add model specific metrics
                    tqdm_metrics = self.__training_tqdm_dict
                    self.progress_bar.set_postfix(**tqdm_metrics)

        # activate batch end hook
        if self.__is_function_implemented('on_batch_end'):
            model = self.__get_model()
            model.on_batch_end()

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics = {k: v for d in all_callback_metrics for k, v in d.items()}

        return 0, grad_norm_dic, all_log_metrics

    def __run_evaluation(self, test=False):
        # when testing make sure user defined a test step
        can_run_test_step = False
        if test:
            can_run_test_step = self.__is_overriden('test_step') and self.__is_overriden('test_end')
            if not can_run_test_step:
                m = '''You called .test() without defining a test step or test_end.
                Please define and try again'''
                raise MisconfigurationException(m)

        # validate only if model has validation_step defined
        # test only if test_step or validation_step are defined
        run_val_step = self.__is_overriden('validation_step')

        if run_val_step or can_run_test_step:

            # hook
            model = self.__get_model()
            model.on_pre_performance_check()

            # select dataloaders
            dataloaders = self.get_val_dataloaders()
            max_batches = self.nb_val_batches

            # calculate max batches to use
            if test:
                dataloaders = self.get_test_dataloaders()
                max_batches = self.nb_test_batches

            # cap max batches to 1 when using fast_dev_run
            if self.fast_dev_run:
                max_batches = 1

            # run evaluation
            eval_results = self.evaluate(self.model,
                                         dataloaders,
                                         max_batches,
                                         test)
            _, prog_bar_metrics, log_metrics, callback_metrics = self.__process_output(eval_results)

            # add metrics to prog bar
            self.__add_tqdm_metrics(prog_bar_metrics)

            # log metrics
            self.__log_metrics(log_metrics, {})

            # track metrics for callbacks
            self.callback_metrics = callback_metrics

            # hook
            model.on_post_performance_check()

            if self.show_progress_bar:
                # add model specific metrics
                tqdm_metrics = self.__training_tqdm_dict
                self.progress_bar.set_postfix(**tqdm_metrics)

            # reduce learning rate based on metrics
            if self.val_loss_drop_lr_callback is not None and not test:
                self.val_loss_drop_lr_callback.on_epoch_end(epoch=self.current_epoch,
                                                            logs=callback_metrics)

        # model checkpointing
        if self.proc_rank == 0 and self.checkpoint_callback is not None and not test:
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch,
                                                  logs=self.callback_metrics)
=======
            self.run_evaluation(test=True)
>>>>>>> 28c3bcb0c018d9fadf82085939c8392bf2e1fa86
