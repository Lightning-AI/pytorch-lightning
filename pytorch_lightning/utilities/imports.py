import importlib.util

import torch


# XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
# TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None

def is_apex_available():
    # TODO: use importlib
    try:
        from apex import amp
    except ImportError:
        return False
    else:
        return True


def is_horovod_available():
    # TODO: use importlib
    try:
        import horovod.torch as hvd
    except (ModuleNotFoundError, ImportError):
        return False
    else:
        return True


def is_hydra_available():
    # TODO: use importlib
    try:
        from hydra.utils import to_absolute_path, get_original_cwd
        from hydra.core.hydra_config import HydraConfig
    except ImportError:
        return False
    else:
        return True


def is_native_amp_available():
    return hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")


def is_omegaconf_available():
    # TODO: use importlib
    try:
        from omegaconf import Container
    except ImportError:
        return False
    else:
        return True


def is_torchtext_available():
    return importlib.util.find_spec("torchtext") is not None


def is_xla_available():
    # TODO: use importlib
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError:
        return False
    else:
        return True
