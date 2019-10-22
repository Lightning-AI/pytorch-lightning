import torch

from pytorch_lightning.pt_overrides.override_data_parallel import (
    LightningDistributedDataParallel, LightningDataParallel)
from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class TrainerDPMixin(object):
    def copy_trainer_model_properties(self, model):
        if isinstance(model, LightningDataParallel):
            ref_model = model.module
        elif isinstance(model, LightningDistributedDataParallel):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self
            m.on_gpu = self.on_gpu
            m.use_dp = self.use_dp
            m.use_ddp2 = self.use_ddp2
            m.use_ddp = self.use_ddp
            m.use_amp = self.use_amp
            m.testing = self.testing
            m.single_gpu = self.single_gpu

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

    def single_gpu_train(self, model):
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

        self.run_pretrain_routine(model)

    def dp_train(self, model):

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

        self.run_pretrain_routine(model)


def normalize_parse_gpu_string_input(s):
    if type(s) is str:
        if s == '-1':
            return -1
        else:
            return [int(x.strip()) for x in s.split(',')]
    else:
        return s


def get_all_available_gpus():
    """
    :return: a list of all available gpus
    """
    return list(range(torch.cuda.device_count()))


def check_gpus_data_type(gpus):
    """
    :param gpus: gpus parameter as passed to the Trainer
        Function checks that it is one of: None, Int, String or List
        Throws otherwise
    :return: return unmodified gpus variable
    """

    if (gpus is not None and
        type(gpus) is not int and
        type(gpus) is not str and
        type(gpus) is not list):    # noqa E129
        raise MisconfigurationException("GPUs must be int, string or list of ints or None.")


def normalize_parse_gpu_input_to_list(gpus):
    assert gpus is not None
    if isinstance(gpus, list):
        return gpus
    else:  # must be an int
        if not gpus:  # gpus==0
            return None
        elif gpus == -1:
            return get_all_available_gpus()
        else:
            return list(range(gpus))


def sanitize_gpu_ids(gpus):
    """
    :param gpus: list of ints corresponding to GPU indices
        Checks that each of the GPUs in the list is actually available.
        Throws if any of the GPUs is not available.
    :return: unmodified gpus variable
    """
    all_available_gpus = get_all_available_gpus()
    for gpu in gpus:
        if gpu not in all_available_gpus:
            message = f"""
            Non-available gpu index {gpu} specified:
            Available gpu indices are: {all_available_gpus}
            """
            raise MisconfigurationException(message)
    return gpus


def parse_gpu_ids(gpus):
    """
    :param gpus: Int, string or list
        An int -1 or string '-1' indicate that all available GPUs should be used.
        A list of ints or a string containing list of comma separated integers
        indicates specific GPUs to use
        An int 0 means that no GPUs should be used
        Any int N > 0 indicates that GPUs [0..N) should be used.
    :return: List of gpus to be used

        If no GPUs are available but the value of gpus variable indicates request for GPUs
        then a misconfiguration exception is raised.
    """

    # Check that gpus param is None, Int, String or List
    check_gpus_data_type(gpus)

    # Handle the case when no gpus are requested
    if gpus is None or type(gpus) is int and gpus == 0:
        return None

    # We know user requested GPUs therefore if some of the
    # requested GPUs are not available an exception is thrown.

    gpus = normalize_parse_gpu_string_input(gpus)
    gpus = normalize_parse_gpu_input_to_list(gpus)
    gpus = sanitize_gpu_ids(gpus)

    if not gpus:
        raise MisconfigurationException("GPUs requested but non are available.")
    return gpus


def determine_root_gpu_device(gpus):
    """
    :param gpus: non empty list of ints representing which gpus to use
    :return: designated root GPU device
    """
    if gpus is None:
        return None

    assert isinstance(gpus, list), "gpus should be a list"
    assert len(gpus), "gpus should be a non empty list"

    # set root gpu
    root_gpu = gpus[0]

    return root_gpu
