"""
Root module for all distributed operations in Lightning.
Currently supports training on CPU, GPU (dp, ddp, ddp2, horovod) and TPU.

"""

from contextlib import ExitStack
import os
from abc import ABC, abstractmethod
import time
import random
import torch
from typing import Union, Callable, Any, List, Optional

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.data import transfer_data_to_device

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except ImportError:
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class TrainerDPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    use_dp: bool
    use_ddp2: bool
    use_ddp: bool
    testing: bool
    single_gpu: bool
    root_gpu: ...
    amp_level: str
    precision: ...
    proc_rank: int
    tpu_local_core_rank: int
    tpu_global_core_rank: int
    use_tpu: bool
    use_native_amp: bool
    data_parallel_device_ids: ...
    logger: Union[LightningLoggerBase, bool]
    progress_bar_callback: ...
    tpu_id: int

    @property
    @abstractmethod
    def use_amp(self) -> bool:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def run_pretrain_routine(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def init_optimizers(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def copy_trainer_model_properties(self, model):
        if isinstance(model, LightningDataParallel):
            ref_model = model.module
        elif isinstance(model, LightningDistributedDataParallel):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self
            m.use_dp = self.use_dp
            m.use_ddp2 = self.use_ddp2
            m.use_ddp = self.use_ddp
            m.use_amp = self.use_amp
            m.testing = self.testing
            m.single_gpu = self.single_gpu
            m.use_tpu = self.use_tpu
            m.tpu_local_core_rank = self.tpu_local_core_rank
            m.tpu_global_core_rank = self.tpu_global_core_rank

    def transfer_batch_to_tpu(self, batch: Any, tpu_id: Optional[int] = None):
        """
        Transfers the data to the TPU.

        Args:
            batch: A tensor or collection of tensors.
            tpu_id: The id of the TPU core. If omitted, the first available core is chosen.

        Return:
            the tensor on the TPU device.

        See Also:
            - :func:`~pytorch_lightning.utilities.apply_func.move_data_to_device`
        """
        if not XLA_AVAILABLE:
            raise MisconfigurationException(
                'Requested to transfer batch to TPU but XLA is not available.'
                ' Are you sure this machine has TPUs?'
            )
        device = xm.xla_device(tpu_id)
        return self.__transfer_batch_to_device(batch, device)

    def transfer_batch_to_gpu(self, batch: Any, gpu_id: Optional[int] = None):
        """
        Transfers the data to the GPU.

        Args:
            batch: A tensor or collection of tensors.
            gpu_id: The id of the GPU device. If omitted, the first available GPU is chosen.

        Return:
            the tensor on the GPU device.

        See Also:
            - :func:`~pytorch_lightning.utilities.apply_func.move_data_to_device`
        """
        device = torch.device('cuda', gpu_id)
        return self.__transfer_batch_to_device(batch, device)

    def __transfer_batch_to_device(self, batch: Any, device: torch.device):
        model = self.get_model()
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

    def single_gpu_train(self, model):
        model.cuda(self.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers, self.optimizer_frequencies = self.init_optimizers(model)

        # TODO: update for 0.8.0
        if self.use_amp and not self.use_native_amp:
            # An example
            model, optimizers = model.configure_apex(amp, model, self.optimizers, self.amp_level)
            self.optimizers = optimizers
            self.reinit_scheduler_properties(self.optimizers, self.lr_schedulers)

        self.run_pretrain_routine(model)

    def tpu_train(self, tpu_core_idx, model):
        # put model on tpu
        self._device = xm.xla_device(self.tpu_id) if self.tpu_id is not None else xm.xla_device()
        model.to(self._device)

        # get the appropriate tpu ranks
        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

        # avoid duplicating progress bar
        if self.tpu_global_core_rank != 0 and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        self.proc_rank = self.tpu_local_core_rank
        rank_zero_only.rank = self.proc_rank

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers, self.optimizer_frequencies = self.init_optimizers(model)

        # init 16 bit for TPU
        if self.precision == 16:
            os.environ['XLA_USE_BF16'] = str(1)

        log.info(f'INIT TPU local core: {self.tpu_local_core_rank},'
                 f' global rank: {self.tpu_global_core_rank}')

        # continue training routine
        self.run_pretrain_routine(model)

        # when training ends on these platforms dump weights to get out of the main process
        if self.on_colab_kaggle:
            self.save_spawn_weights(model)

    def dp_train(self, model):

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers, self.optimizer_frequencies = self.init_optimizers(model)

        model.cuda(self.root_gpu)

        # hack forward to do autocast for the user
        model_autocast_original_forward = model.forward
        if self.use_amp and self.use_native_amp:
            # wrap the user's forward in autocast and give it back at the end
            model.forward = torch.cuda.amp.autocast()(model.forward)

        # TODO: remove in v0.8.0
        # check for this bug (amp + dp + !01 doesn't work)
        # https://github.com/NVIDIA/apex/issues/227
        if self.use_dp and self.use_amp and not self.use_native_amp:
            if self.amp_level == 'O2':
                raise MisconfigurationException(
                    f'Amp level {self.amp_level} with DataParallel is not supported.'
                    f' See this note from NVIDIA for more info: https://github.com/NVIDIA/apex/issues/227.'
                    f' We recommend you switch to ddp if you want to use amp')
            else:
                model, optimizers = model.configure_apex(amp, model, self.optimizers, self.amp_level)
                self.reinit_scheduler_properties(optimizers, self.lr_schedulers)

        # create list of device ids
        device_ids = self.data_parallel_device_ids
        if isinstance(device_ids, int):
            device_ids = list(range(device_ids))

        # set dp device
        torch.cuda.set_device(self.root_gpu)

        model = LightningDataParallel(model, device_ids=device_ids)

        self.run_pretrain_routine(model)

        model.forward = model_autocast_original_forward

    def horovod_train(self, model):
        if torch.cuda.is_available() and self.on_gpu:
            # Horovod: pin GPU to local rank
            assert self.root_gpu == hvd.local_rank()
            torch.cuda.set_device(self.root_gpu)
            model.cuda(self.root_gpu)

        # avoid duplicating progress bar
        if hvd.rank() != 0 and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers, self.optimizer_frequencies = self.init_optimizers(model)

        # Horovod: scale the learning rate by the number of workers to account for
        # increased total batch size
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= hvd.size()

        if self.use_amp:
            # An example
            model, optimizers = model.configure_apex(amp, model, self.optimizers, self.amp_level)
            self.optimizers = optimizers
            self.reinit_scheduler_properties(self.optimizers, self.lr_schedulers)

        # Horovod: broadcast parameters & optimizer state to ensure consistent initialization
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        for optimizer in self.optimizers:
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        def filter_named_parameters(model, optimizer):
            opt_params = set([p for group in optimizer.param_groups for p in group.get('params', [])])
            return [(name, p) for name, p in model.named_parameters() if p in opt_params]

        # Horovod: wrap optimizers to perform gradient aggregation via allreduce
        self.optimizers = [
            hvd.DistributedOptimizer(optimizer, named_parameters=filter_named_parameters(model, optimizer))
            for optimizer in self.optimizers
        ]

        # Update logger rank info from Horovod to avoid race conditions from  different ranks
        # creating directories / writing files in the same locations.
        self.proc_rank = hvd.rank()
        rank_zero_only.rank = self.proc_rank

        with ExitStack() as stack:
            for optimizer in self.optimizers:
                # Synchronization will be performed explicitly following backward()
                stack.enter_context(optimizer.skip_synchronize())

            self.run_pretrain_routine(model)

        # Make sure all workers have finished training before returning to the user
        hvd.join()


def normalize_parse_gpu_string_input(s):
    if isinstance(s, str):
        if s == '-1':
            return -1
        else:
            return [int(x.strip()) for x in s.split(',') if len(x) > 0]
    else:
        return s


def get_all_available_gpus() -> List[int]:
    """
    Returns:
         a list of all available gpus
    """
    return list(range(torch.cuda.device_count()))


def check_gpus_data_type(gpus: Any) -> None:
    """
    Checks that the gpus argument is one of: None, Int, String or List.
    Raises a MisconfigurationException otherwise.

    Args:
        gpus: parameter as passed to the Trainer
    """
    if gpus is not None and (not isinstance(gpus, (int, str, list)) or isinstance(gpus, bool)):
        raise MisconfigurationException("GPUs must be int, string or list of ints or None.")


def normalize_parse_gpu_input_to_list(gpus: Union[int, List[int]]) -> Optional[List[int]]:
    assert gpus is not None
    if isinstance(gpus, list):
        return gpus

    # must be an int
    if not gpus:  # gpus==0
        return None
    if gpus == -1:
        return get_all_available_gpus()

    return list(range(gpus))


def sanitize_gpu_ids(gpus: List[int]) -> List[int]:
    """
    Checks that each of the GPUs in the list is actually available.
    Raises a MisconfigurationException if any of the GPUs is not available.

    Args:
        gpus: list of ints corresponding to GPU indices

    Returns:
        unmodified gpus variable
    """
    all_available_gpus = get_all_available_gpus()
    misconfig = False
    for gpu in gpus:
        if gpu not in all_available_gpus:
            misconfig = True

    if misconfig:
        # sometimes auto ddp might have different flags
        # but this is not what the user intended
        # correct for the user
        if len(gpus) == len(all_available_gpus):
            gpus = all_available_gpus
        else:
            raise MisconfigurationException(f"""
                You requested GPUs: {gpus}
                But your machine only has: {all_available_gpus}
            """)
    return gpus


def parse_gpu_ids(gpus: Union[int, str, List]) -> Optional[List[int]]:
    """
    Parses the GPU ids given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        gpus: An int -1 or string '-1' indicate that all available GPUs should be used.
            A list of ints or a string containing list of comma separated integers
            indicates specific GPUs to use.
            An int 0 means that no GPUs should be used.
            Any int N > 0 indicates that GPUs [0..N) should be used.

    Returns:
        a list of gpus to be used or ``None`` if no GPUs were requested

    If no GPUs are available but the value of gpus variable indicates request for GPUs
    then a MisconfigurationException is raised.
    """

    # nothing was passed into the GPUs argument
    if callable(gpus):
        return None

    # Check that gpus param is None, Int, String or List
    check_gpus_data_type(gpus)

    # Handle the case when no gpus are requested
    if gpus is None or isinstance(gpus, int) and gpus == 0:
        return None

    # We know user requested GPUs therefore if some of the
    # requested GPUs are not available an exception is thrown.

    gpus = normalize_parse_gpu_string_input(gpus)
    gpus = normalize_parse_gpu_input_to_list(gpus)
    gpus = sanitize_gpu_ids(gpus)

    if not gpus:
        raise MisconfigurationException("GPUs requested but none are available.")
    return gpus


def determine_root_gpu_device(gpus: List[int]) -> Optional[int]:
    """
    Args:
        gpus: non-empty list of ints representing which gpus to use

    Returns:
        designated root GPU device id
    """
    if gpus is None:
        return None

    assert isinstance(gpus, list), "gpus should be a list"
    assert len(gpus) > 0, "gpus should be a non empty list"

    # set root gpu
    root_gpu = gpus[0]

    return root_gpu


def retry_jittered_backoff(func: Callable, num_retries: int = 5, cap_delay: float = 1.0, base_delay: float = 0.01):
    """Retry jittered backoff.

    Based on:
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

    Args:
        func: tested function
        num_retries: number of tries
        cap_delay: max sleep time
        base_delay: initial sleep time is 10ms
    """
    sleep_delay = base_delay         # initial sleep time is 10ms

    for i in range(num_retries):
        try:
            return func()
        except RuntimeError as err:
            if i == num_retries - 1:
                raise err
            else:
                continue
        time.sleep(sleep_delay)
        sleep_delay = min(cap_delay, random.uniform(base_delay, sleep_delay * 3))


def pick_single_gpu(exclude_gpus: list):
    for i in range(torch.cuda.device_count()):
        if i in exclude_gpus:
            continue
        # Try to allocate on device:
        device = torch.device(f"cuda:{i}")
        try:
            torch.ones(1).to(device)
        except RuntimeError:
            continue
        return i
    raise RuntimeError("No GPUs available.")


def pick_multiple_gpus(nb):
    picked = []
    for _ in range(nb):
        picked.append(pick_single_gpu(exclude_gpus=picked))

    return picked
