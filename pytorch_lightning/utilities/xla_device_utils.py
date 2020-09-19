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
    except Exception as _e:
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


class XLADeviceUtils:
    """Used to detect the type of XLA device"""

    TPU_AVAILABLE = None

    @staticmethod
    def _fetch_xla_device_type(device):
        """ Returns XLA device type """
        if xm is not None:
            return xm.xla_device_hw(device)

    @staticmethod
    def _is_device_tpu():
        """ Check if device is TPU """
        if xm is not None:
            device = xm.xla_device()
            device_type = XLADeviceUtils._fetch_xla_device_type(device)
            return device_type == "TPU"

    @staticmethod
    def tpu_device_exists():
        """ Public method to check if TPU is available """
        if XLADeviceUtils.TPU_AVAILABLE is None and TORCHXLA_AVAILABLE:
            XLADeviceUtils.TPU_AVAILABLE = pl_multi_process(XLADeviceUtils._is_device_tpu)()
        return XLADeviceUtils.TPU_AVAILABLE
