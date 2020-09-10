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

import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as torch_distrib
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.step_result import EvalResult
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler, PassThroughProfiler, SimpleProfiler
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.configuration_validator import ConfigValidator
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import TrainerDeprecatedAPITillVer0_10
from pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.states import TrainerState, trainer_state
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.utilities import parsing, rank_zero_info, rank_zero_only, rank_zero_warn, AMPType
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.trainer.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.accelerators.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.logger_connector import LoggerConnector
from pytorch_lightning.trainer.lr_scheduler_connector import LRSchedulerConnector
from pytorch_lightning.trainer.training_trick_connector import TrainingTricksConnector
from pytorch_lightning.trainer.callback_connector import CallbackConnector
from pytorch_lightning.trainer.model_connector import ModelConnector
from pytorch_lightning.trainer.debugging_connector import DebuggingConnector
from pytorch_lightning import _logger as log
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.trainer.precision_connector import PrecisionConnector
from pytorch_lightning.trainer.data_connector import DataConnector
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.trainer import docstrings
from pytorch_lightning.trainer.properties import TrainerProperties

# warnings to ignore in trainer
warnings.filterwarnings(
    'ignore', message='torch.distributed.reduce_op is deprecated, ' 'please use torch.distributed.ReduceOp instead'
)

try:
    from apex import amp
except ImportError:
    amp = None

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class Trainer(
    TrainerProperties,
    TrainerIOMixin,
    TrainerCallbackHookMixin,
    TrainerModelHooksMixin,
    TrainerOptimizersMixin,
    TrainerDDPMixin,
    TrainerLoggingMixin,
    TrainerTrainingTricksMixin,
    TrainerDataLoadingMixin,
    TrainerDeprecatedAPITillVer0_10,
):
    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: Union[ModelCheckpoint, bool] = True,
        early_stop_callback: Optional[Union[EarlyStopping, bool]] = False,
        callbacks: Optional[List[Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0,
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: int = 1,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: bool = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: int = 1000,
        min_epochs: int = 1,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        log_save_interval: int = 100,
        row_log_interval: int = 50,
        distributed_backend: Optional[str] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = ModelSummary.MODE_DEFAULT,
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        profiler: Optional[Union[BaseProfiler, bool]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        amp_backend: str = 'native',
        amp_level: str = 'O2',  # backward compatible, todo: remove in v1.0.0
        val_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        test_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        train_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        overfit_pct: float = None,  # backward compatible, todo: remove in v1.0.0
    ):
        super().__init__()

        self.deterministic = deterministic
        torch.backends.cudnn.deterministic = self.deterministic
        if self.deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)

        # init the default rank if exists
        # we need to call this here or NVIDIA flags and other messaging in init will show on all ranks
        # this way we only show it on rank 0
        if 'LOCAL_RANK' in os.environ:
            rank_zero_only.rank = int(os.environ['LOCAL_RANK'])

        # tracks internal state for debugging
        self.dev_debugger = InternalDebugger(self)
        self.config_validator = ConfigValidator(self)
        self.data_connector = DataConnector(self)
        self.lr_scheduler_connector = LRSchedulerConnector(self)
        self.accelerator_connector = AcceleratorConnector(self)
        self.logger_connector = LoggerConnector(self)
        self.model_connector = ModelConnector(self)
        self.precision_connector = PrecisionConnector(self)
        self.callback_connector = CallbackConnector(self)
        self.debugging_connector = DebuggingConnector(self)
        self.training_tricks_connector = TrainingTricksConnector(self)

        self.tuner = Tuner(self)
        self.accelerator_backend = None

        # loops
        self.evaluation_loop = EvaluationLoop(self)
        self.train_loop = TrainLoop(self)

        # training bookeeping
        self.total_batch_idx = 0
        self.batch_idx = 0
        self.num_training_batches = 0
        self.num_val_batches = []
        self.num_sanity_val_batches = []
        self.num_test_batches = []
        self.train_dataloader = None
        self.test_dataloaders = None
        self.val_dataloaders = None

        # when true, prints test results
        self.verbose_test = True

        # when .test() is called, it sets this
        self.tested_ckpt_path = None

        # training state
        self.weights_summary = weights_summary
        self.model = None
        self.datamodule = None
        self.testing = False
        self.prepare_data_per_node = prepare_data_per_node
        self.lr_schedulers = []
        self.optimizers = None
        self.optimizer_frequencies = []
        self.global_step = 0
        self.current_epoch = 0
        self.interrupted = False
        self.should_stop = False
        self.running_sanity_check = False
        self._state = TrainerState.INITIALIZING

        self._default_root_dir = default_root_dir or os.getcwd()
        self._weights_save_path = weights_save_path or self._default_root_dir

        # init callbacks
        self.callback_connector.on_trainer_init(
            callbacks,
            early_stop_callback,
            checkpoint_callback,
            progress_bar_refresh_rate,
            process_position
        )

        # init data flags
        self.data_connector.on_trainer_init(check_val_every_n_epoch, reload_dataloaders_every_epoch)

        # hook
        self.on_init_start()

        # init training tricks
        self.training_tricks_connector.on_trainer_init(gradient_clip_val, track_grad_norm, accumulate_grad_batches)

        # init accelerator related flags
        self.accelerator_connector.on_trainer_init(
            num_processes,
            tpu_cores,
            distributed_backend,
            auto_select_gpus,
            gpus,
            num_nodes,
            log_gpu_memory,
            sync_batchnorm,
            benchmark,
            replace_sampler_ddp
        )

        # init train loop related flags
        self.train_loop.on_init_start(max_epochs, min_epochs, max_steps, min_steps, num_sanity_val_steps)

        self.auto_lr_find = auto_lr_find
        self.auto_scale_batch_size = auto_scale_batch_size
        self._is_data_prepared = False

        self.truncated_bptt_steps = truncated_bptt_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.terminate_on_nan = terminate_on_nan
        self.shown_warnings = set()

        # configure profiler
        if profiler is True:
            profiler = SimpleProfiler()
        self.profiler = profiler or PassThroughProfiler()

        # init logger flags
        self.logger_connector.on_trainer_init(logger, log_save_interval, row_log_interval)

        # init debugging flags
        self.debugging_connector.on_init_start(
            overfit_pct,
            val_percent_check,
            test_percent_check,
            train_percent_check,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            val_check_interval,
            overfit_batches,
            fast_dev_run
        )

        # set precision
        self.precision_connector.on_trainer_init(precision, amp_level, amp_backend)

        # Callback system
        self.on_init_end()

    def tune(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        # TODO: temporary, need to decide if tune or separate object

        # setup data, etc...
        self.setup_fit(model, train_dataloader, val_dataloaders, datamodule)

        # hook
        self.call_hook('on_fit_start', model)

        # hook
        self.data_connector.prepare_data(model)

        # Run auto batch size scaling
        if self.auto_scale_batch_size:
            if isinstance(self.auto_scale_batch_size, bool):
                self.auto_scale_batch_size = 'power'
            self.tuner.scale_batch_size(
                model,
                mode=self.auto_scale_batch_size,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
            )
            model.logger = self.logger  # reset logger binding

        # Run learning rate finder:
        if self.auto_lr_find:
            self.tuner.internal_find_lr(self, model)
            model.logger = self.logger  # reset logger binding

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    @trainer_state(entering=TrainerState.RUNNING, exiting=TrainerState.FINISHED)
    def fit(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        results = None

        # setup data, etc...
        self.setup_fit(model, train_dataloader, val_dataloaders, datamodule)

        # hook
        self.call_hook('on_fit_start', model)

        # hook
        self.data_connector.prepare_data(model)

        # set testing if set in environ
        self.testing = os.environ.get('PL_TESTING_MODE', self.testing)

        # -------------------------
        # TRAIN
        # -------------------------
        self.accelerator_backend = self.accelerator_connector.select_accelerator()
        self.accelerator_backend.setup(model)
        results = self.accelerator_backend.train()
        self.accelerator_backend.teardown()

        # -------------------------
        # POST-Training
        # -------------------------
        # hook
        self.call_hook('on_fit_end')

        # hook
        self.teardown('fit')
        if self.is_function_implemented('teardown'):
            model.teardown('fit')

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return results or 1

    def setup_fit(self, model, train_dataloader, val_dataloaders, datamodule):
        # bind logger and other properties
        self.model_connector.copy_trainer_model_properties(model)

        # clean hparams
        if hasattr(model, 'hparams'):
            parsing.clean_namespace(model.hparams)

        # links data to the trainer
        self.data_connector.attach_data(model, train_dataloader, val_dataloaders, datamodule)

        # check that model is configured correctly
        self.config_validator.verify_loop_configurations(model)

    def setup_training(self, model: LightningModule):
        """Sanity check a few things before starting actual training.

        Args:
            model: The model to run sanity test on.
        """
        # --------------------------
        # Setup??
        # --------------------------
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.model_connector.copy_trainer_model_properties(ref_model)

        # init amp. Must be done here instead of __init__ to allow ddp to work
        if self.amp_backend == AMPType.NATIVE and self.precision == 16 and not self.use_tpu:
            self.scaler = torch.cuda.amp.GradScaler()

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started
            self.logger.log_hyperparams(ref_model.hparams)
            self.logger.log_graph(ref_model)
            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            torch_distrib.barrier()

        # wait for all models to restore weights
        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous("pl.Trainer.setup_training")

        elif self.use_horovod:
            # wait for all processes to catch up
            hvd.join()

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # --------------------------
        # Pre-train
        # --------------------------
        # on pretrain routine start
        self.on_pretrain_routine_start(ref_model)
        if self.is_function_implemented('on_pretrain_routine_start'):
            ref_model.on_pretrain_routine_start()

        # print model summary
        if self.is_global_zero and self.weights_summary is not None and not self.testing:
            if self.weights_summary in ModelSummary.MODES:
                ref_model.summarize(mode=self.weights_summary)
            else:
                raise MisconfigurationException("weights_summary can be None, " + ", ".join(ModelSummary.MODES))

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc is called
        self.restore_weights(model)

        # on pretrain routine end
        self.on_pretrain_routine_end(ref_model)
        if self.is_function_implemented('on_pretrain_routine_end'):
            ref_model.on_pretrain_routine_end()

    def train(self):
        self.run_sanity_check(self.get_model())

        # enable train mode
        model = self.get_model()
        model.train()
        torch.set_grad_enabled(True)

        # reload data when needed
        self.train_loop.reset_train_val_dataloaders(model)

        # hook
        self.train_loop.on_train_start()

        try:
            # run all epochs
            for epoch in range(self.current_epoch, self.max_epochs):

                # reset train dataloader
                if self.reload_dataloaders_every_epoch:
                    self.reset_train_dataloader(model)

                # hook
                self.train_loop.on_train_epoch_start(epoch)

                # run train epoch
                self.train_loop.run_training_epoch()

                if self.max_steps and self.max_steps <= self.global_step:

                    # hook
                    self.train_loop.on_train_end()
                    return

                # update LR schedulers
                self.lr_scheduler_connector.update_learning_rates(interval='epoch')

                # early stopping
                met_min_epochs = epoch >= self.min_epochs - 1
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                if self.should_stop:
                    if (met_min_epochs and met_min_steps):
                        self.train_loop.on_train_end()
                        return
                    else:
                        log.info('Trainer was signaled to stop but required minimum epochs'
                                 f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                                 ' not been met. Training will continue...')

            # hook
            self.train_loop.on_train_end()

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')

            # user could press ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.interrupted = True
                self._state = TrainerState.INTERRUPTED
                self.on_keyboard_interrupt()

                # hook
                self.train_loop.on_train_end()

    def run_evaluation(self, test_mode: bool = False, max_batches=None):
        # bookkeeping
        self.evaluation_loop.testing = test_mode
        dataloaders, max_batches = self.evaluation_loop.get_evaluation_dataloaders(max_batches)
        if self.evaluation_loop.should_skip_evaluation(dataloaders, max_batches):
            return [], []

        # enable eval mode + no grads
        model = self.get_model()
        model.zero_grad()
        model.eval()
        torch.set_grad_enabled(False)

        # hook
        self.evaluation_loop.on_evaluation_start()

        # set up the eval loop
        self.evaluation_loop.setup(model, max_batches, dataloaders)

        # hook
        # TODO: should this be insider the dataloader loop?
        self.evaluation_loop.on_evaluation_epoch_start()

        # run validation/testing
        for dataloader_idx, dataloader in enumerate(dataloaders):
            # bookkeeping
            dl_outputs = []
            dataloader = self.accelerator_backend.process_dataloader(dataloader)
            dl_max_batches = self.evaluation_loop.max_batches[dataloader_idx]

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when running on limited batches
                if batch_idx >= dl_max_batches:
                    break

                # hook
                self.evaluation_loop.on_evaluation_batch_start(batch, batch_idx, dataloader_idx)

                # lightning module methods
                output = self.evaluation_loop.evaluation_step(test_mode, batch, batch_idx, dataloader_idx)
                output = self.evaluation_loop.evaluation_step_end(output)

                # hook
                self.evaluation_loop.on_evaluation_batch_end(batch, batch_idx, dataloader_idx)

                # clean up
                self.evaluation_loop.evaluation_batch_end_cleanup(output, batch_idx, dataloader_idx)
                self.evaluation_loop.log_step_metrics(output, batch_idx)

                # track epoch level metrics
                if output is not None:
                    dl_outputs.append(output)

            self.evaluation_loop.outputs.append(dl_outputs)

        # lightning module method
        eval_results = self.evaluation_loop.evaluation_epoch_end(num_dataloaders=len(dataloaders))

        # bookkeeping
        eval_loop_results = self.evaluation_loop.log_epoch_metrics(eval_results, test_mode)
        self.evaluation_loop.predictions.to_disk()

        # hook
        self.evaluation_loop.on_evaluation_epoch_end()

        # enable train mode again
        model.train()
        torch.set_grad_enabled(True)

        # hook
        self.evaluation_loop.on_evaluation_end()

        return eval_loop_results, eval_results

    def run_test(self):
        # only load test dataloader for testing
        # self.reset_test_dataloader(ref_model)
        eval_loop_results, _ = self.run_evaluation(test_mode=True)

        if len(eval_loop_results) == 0:
            return 1

        # remove the tensors from the eval results
        for i, result in enumerate(eval_loop_results):
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.cpu().item()

        return eval_loop_results

    def train_or_test(self):
        if self.testing:
            results = self.run_test()
        else:
            results = self.train()
        return results

    def run_sanity_check(self, ref_model):
        using_val_step = ref_model.val_dataloader is not None and is_overridden('validation_step', ref_model)
        should_sanity_check = using_val_step and self.num_sanity_val_steps > 0 and self.limit_val_batches > 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            self.reset_val_dataloader(ref_model)
            self.num_sanity_val_batches = [
                min(self.num_sanity_val_steps, val_batches) for val_batches in self.num_val_batches
            ]

            # hook and callback
            self.running_sanity_check = True
            self.on_sanity_check_start()

            # run eval step
            _, eval_results = self.run_evaluation(test_mode=False, max_batches=self.num_sanity_val_batches)

            # allow no returns from eval
            if eval_results is not None and len(eval_results) > 0:
                # when we get a list back, used only the last item
                if isinstance(eval_results, list):
                    eval_results = eval_results[-1]

                if isinstance(eval_results, EvalResult):
                    callback_metrics = eval_results.callback_metrics
                else:
                    _, _, _, callback_metrics, _ = self.process_output(eval_results)
                self.logger_connector.callback_metrics = callback_metrics

            self.on_sanity_check_end()
            self.running_sanity_check = False

    @trainer_state(entering=TrainerState.RUNNING, exiting=TrainerState.FINISHED)
    def test(
        self,
        model: Optional[LightningModule] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ):
        # --------------------
        # SETUP HOOK
        # --------------------
        self.verbose_test = verbose

        if self.global_rank != 0:
            return

        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if test_dataloaders and datamodule:
            raise MisconfigurationException(
                'You cannot pass test_dataloaders to trainer.test if you supply a datamodule'
            )

        # Attach datamodule to get setup/prepare_data added to model before the call to it below
        self.data_connector.attach_datamodule(model or self.get_model(), datamodule, 'test')

        if model is not None:
            results = self.__test_given_model(model, test_dataloaders)
        else:
            results = self.__test_using_best_weights(ckpt_path, test_dataloaders)

        self.teardown('test')

        return results

    def __test_using_best_weights(self, ckpt_path, test_dataloaders):
        model = self.get_model()

        # if user requests the best checkpoint but we don't have it, error
        if ckpt_path == 'best' and self.checkpoint_callback.save_top_k <= 0:
            raise MisconfigurationException(
                'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.'
            )

        # load best weights
        if ckpt_path is not None:
            # ckpt_path is 'best' so load the best model
            if ckpt_path == 'best':
                ckpt_path = self.checkpoint_callback.best_model_path

            if len(ckpt_path) == 0:
                rank_zero_warn(
                    f'.test() found no path for the best weights, {ckpt_path}. Please '
                    f'specify a path for a checkpoint .test(ckpt_path=PATH)'
                )
                return {}

            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['state_dict'])

        # attach dataloaders
        if test_dataloaders is not None:
            self.data_connector.attach_dataloaders(model, test_dataloaders=test_dataloaders)

        # run tests
        self.tested_ckpt_path = ckpt_path
        self.testing = True
        os.environ['PL_TESTING_MODE'] = '1'
        self.model = model
        results = self.fit(model)
        self.testing = False
        del os.environ['PL_TESTING_MODE']

        # teardown
        if self.is_function_implemented('teardown'):
            model_ref = self.get_model()
            model_ref.teardown('test')

        return results

    def __test_given_model(self, model, test_dataloaders):

        # attach data
        if test_dataloaders is not None:
            self.data_connector.attach_dataloaders(model, test_dataloaders=test_dataloaders)

        # run test
        # sets up testing so we short circuit to eval
        self.testing = True
        self.model = model
        results = self.fit(model)
        self.testing = False

        # teardown
        if self.is_function_implemented('teardown'):
            model.teardown('test')

        return results

    def call_setup_hook(self, model):
        # call setup after the ddp process has connected
        stage_name = 'test' if self.testing else 'fit'
        if self.datamodule is not None:
            called = self.datamodule.has_setup_test if self.testing else self.datamodule.has_setup_fit
            if not called:
                self.datamodule.setup(stage_name)
        self.setup(stage_name)
        model.setup(stage_name)

    def call_hook(self, hook_name, *args, **kwargs):
        # always profile hooks
        with self.profiler.profile(hook_name):

            # first call trainer hook
            if hasattr(self, hook_name):
                trainer_hook = getattr(self, hook_name)
                trainer_hook(*args, **kwargs)

            # next call hook in lightningModule
            output = None
            model_ref = self.get_model()
            if is_overridden(hook_name, model_ref):
                hook_fx = getattr(model_ref, hook_name)
                output = hook_fx(*args, **kwargs)

            # if the PL module doesn't have the hook then call the accelator
            # used to auto-reduce things for the user with Results obj
            elif hasattr(self.accelerator_backend, hook_name):
                accelerator_hook = getattr(self.accelerator_backend, hook_name)
                output = accelerator_hook(*args, **kwargs)

            return output

# add docstrings
Trainer.__init__.__doc__ = docstrings.trainer.init
Trainer.fit.__doc__ = docstrings.trainer.fit
Trainer.test.__doc__ = docstrings.trainer.test
