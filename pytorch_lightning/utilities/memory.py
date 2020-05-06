import gc
import torch


def recursive_detach(in_dict: dict) -> dict:
    """Detach all tensors in `in_dict`.

    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `torch.Tensor`. Other types in `in_dict` are
    not affected by this utility function.

    Args:
        in_dict:

    Return:
        out_dict:
    """
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, dict):
            out_dict.update({k: recursive_detach(v)})
        elif callable(getattr(v, 'detach', None)):
            out_dict.update({k: v.detach()})
        else:
            out_dict.update({k: v})
    return out_dict


def is_oom_error(exception):
    return is_cuda_out_of_memory(exception) \
        or is_cudnn_snafu(exception) \
        or is_out_of_cpu_memory(exception)


def is_cuda_out_of_memory(exception):
    return isinstance(exception, RuntimeError) \
        and len(exception.args) == 1 \
        and "CUDA out of memory." in exception.args[0]


def is_cudnn_snafu(exception):
    return isinstance(exception, RuntimeError) \
        and len(exception.args) == 1 \
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]


def is_out_of_cpu_memory(exception):
    return isinstance(exception, RuntimeError) \
        and len(exception.args) == 1 \
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]


def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
