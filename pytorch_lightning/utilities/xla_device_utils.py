import functools
import importlib
from multiprocessing import Process, Queue

TORCHXLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
if TORCHXLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
else:
    xm = None


def inner_f(queue, func, **kwargs):
    try:
        queue.put(func(**kwargs))
    except Exception:
        import traceback
        traceback.print_exc()
        queue.put(None)


def pl_multi_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        queue = Queue()
        proc = Process(target=inner_f, args=(queue, func,), kwargs=kwargs)
        proc.start()
        proc.join()
        return queue.get()

    return wrapper


def fetch_xla_device_type(device):
    if xm is not None:
        return xm.xla_device_hw(device)
    else:
        return None


def is_device_tpu():
    if xm is not None:
        device = xm.xla_device()
        device_type = fetch_xla_device_type(device)
        return device_type == "TPU"
    else:
        return False


def tpu_device_exists():
    return pl_multi_process(is_device_tpu)()
