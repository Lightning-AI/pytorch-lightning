import torch


def is_apex_available():
    try:
        from apex import amp
    except ImportError:
        return False
    else:
        return True


def is_hydra_available():
    try:
        from hydra.utils import to_absolute_path, get_original_cwd
        from hydra.core.hydra_config import HydraConfig
    except ImportError:
        return False
    else:
        return True


def is_native_amp_available():
    return hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")


def is_xla_available():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError:
        return False
    else:
        return True
