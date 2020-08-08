import importlib.util

import torch


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_horovod_available():
    return (importlib.util.find_spec("horovod") is not None and
            importlib.util.find_spec("horovod.torch") is not None)


def is_hydra_available():
    return (importlib.util.find_spec("hydra") is not None and
            importlib.util.find_spec("hydra.core") is not None)


def is_native_amp_available():
    return hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")


def is_omegaconf_available():
    return importlib.util.find_spec("omegaconf") is not None


def is_torchtext_available():
    return importlib.util.find_spec("torchtext") is not None


def is_torchvision_available():
    return importlib.util.find_spec("torchvision") is not None


def is_xla_available():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError:
        return False
    else:
        return True
