import multiprocessing
from typing import Any, List, MutableSequence, Optional, Tuple, Union

import torch

from lightning_lite.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from lightning_lite.utilities.exceptions import MisconfigurationException
from lightning_lite.utilities.types import _DEVICE


def determine_root_gpu_device(gpus: List[_DEVICE]) -> Optional[_DEVICE]:
    """
    Args:
        gpus: Non-empty list of ints representing which GPUs to use

    Returns:
        Designated root GPU device id

    Raises:
        TypeError:
            If ``gpus`` is not a list
        AssertionError:
            If GPU list is empty
    """
    if gpus is None:
        return None

    if not isinstance(gpus, list):
        raise TypeError("GPUs should be a list")

    assert len(gpus) > 0, "GPUs should be a non-empty list"

    # set root gpu
    root_gpu = gpus[0]

    return root_gpu


def parse_gpu_ids(
    gpus: Optional[Union[int, str, List[int]]],
    include_cuda: bool = False,
    include_mps: bool = False,
) -> Optional[List[int]]:
    """
    Parses the GPU IDs given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        gpus: An int -1 or string '-1' indicate that all available GPUs should be used.
            A list of unique ints or a string containing a list of comma separated unique integers
            indicates specific GPUs to use.
            An int of 0 means that no GPUs should be used.
            Any int N > 0 indicates that GPUs [0..N) should be used.
        include_cuda: A boolean value indicating whether to include CUDA devices for GPU parsing.
        include_mps: A boolean value indicating whether to include MPS devices for GPU parsing.

    Returns:
        A list of GPUs to be used or ``None`` if no GPUs were requested

    Raises:
        MisconfigurationException:
            If no GPUs are available but the value of gpus variable indicates request for GPUs

    .. note::
        ``include_cuda`` and ``include_mps`` default to ``False`` so that you only
        have to specify which device type to use and all other devices are not disabled.
    """
    # Check that gpus param is None, Int, String or Sequence of Ints
    _check_data_type(gpus)

    # Handle the case when no GPUs are requested
    if gpus is None or (isinstance(gpus, int) and gpus == 0) or str(gpus).strip() in ("0", "[]"):
        return None

    # We know the user requested GPUs therefore if some of the
    # requested GPUs are not available an exception is thrown.
    gpus = _normalize_parse_gpu_string_input(gpus)
    gpus = _normalize_parse_gpu_input_to_list(gpus, include_cuda=include_cuda, include_mps=include_mps)
    if not gpus:
        raise MisconfigurationException("GPUs requested but none are available.")

    if (
        TorchElasticEnvironment.detect()
        and len(gpus) != 1
        and len(_get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)) == 1
    ):
        # Omit sanity check on torchelastic because by default it shows one visible GPU per process
        return gpus

    # Check that GPUs are unique. Duplicate GPUs are not supported by the backend.
    _check_unique(gpus)

    return _sanitize_gpu_ids(gpus, include_cuda=include_cuda, include_mps=include_mps)


def parse_tpu_cores(tpu_cores: Optional[Union[int, str, List[int]]]) -> Optional[Union[int, List[int]]]:
    """
    Parses the tpu_cores given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        tpu_cores: An int of 1 or string '1' indicates that 1 core with multi-processing should be used
            An int 8 or string '8' indicates that all 8 cores with multi-processing should be used
            A list of ints or a strings containing a list of comma separated integers
            indicates the specific TPU core to use.

    Returns:
        A list of tpu_cores to be used or ``None`` if no TPU cores were requested

    Raises:
        MisconfigurationException:
            If TPU cores aren't 1, 8 or [<1-8>]
    """
    _check_data_type(tpu_cores)

    if isinstance(tpu_cores, str):
        tpu_cores = _parse_tpu_cores_str(tpu_cores.strip())

    if not _tpu_cores_valid(tpu_cores):
        raise MisconfigurationException("`tpu_cores` can only be 1, 8 or [<1-8>]")

    return tpu_cores


def parse_cpu_cores(cpu_cores: Union[int, str, List[int]]) -> int:
    """Parses the cpu_cores given in the format as accepted by the ``devices`` argument in the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        cpu_cores: An int > 0.

    Returns:
        An int representing the number of processes

    Raises:
        MisconfigurationException:
            If cpu_cores is not an int > 0
    """
    if isinstance(cpu_cores, str) and cpu_cores.strip().isdigit():
        cpu_cores = int(cpu_cores)

    if not isinstance(cpu_cores, int) or cpu_cores <= 0:
        raise MisconfigurationException("`devices` selected with `CPUAccelerator` should be an int > 0.")

    return cpu_cores


def _normalize_parse_gpu_string_input(s: Union[int, str, List[int]]) -> Union[int, List[int]]:
    if not isinstance(s, str):
        return s
    if s == "-1":
        return -1
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if len(x) > 0]
    return int(s.strip())


def _sanitize_gpu_ids(gpus: List[int], include_cuda: bool = False, include_mps: bool = False) -> List[int]:
    """Checks that each of the GPUs in the list is actually available. Raises a MisconfigurationException if any of
    the GPUs is not available.

    Args:
        gpus: List of ints corresponding to GPU indices

    Returns:
        Unmodified gpus variable

    Raises:
        MisconfigurationException:
            If machine has fewer available GPUs than requested.
    """
    if sum((include_cuda, include_mps)) == 0:
        raise ValueError("At least one gpu type should be specified!")
    all_available_gpus = _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)
    for gpu in gpus:
        if gpu not in all_available_gpus:
            raise MisconfigurationException(
                f"You requested gpu: {gpus}\n But your machine only has: {all_available_gpus}"
            )
    return gpus


def _normalize_parse_gpu_input_to_list(
    gpus: Union[int, List[int], Tuple[int, ...]], include_cuda: bool, include_mps: bool
) -> Optional[List[int]]:
    assert gpus is not None
    if isinstance(gpus, (MutableSequence, tuple)):
        return list(gpus)

    # must be an int
    if not gpus:  # gpus==0
        return None
    if gpus == -1:
        return _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)

    return list(range(gpus))


def _get_all_available_gpus(include_cuda: bool = False, include_mps: bool = False) -> List[int]:
    """
    Returns:
        A list of all available GPUs
    """
    cuda_gpus = _get_all_available_cuda_gpus() if include_cuda else []
    mps_gpus = _get_all_available_mps_gpus() if include_mps else []
    return cuda_gpus + mps_gpus


def _get_all_available_mps_gpus() -> List[int]:
    """
    Returns:
        A list of all available MPS GPUs
    """
    # lazy import to avoid circular dependencies
    # from lightning_lite.accelerators.mps import _MPS_AVAILABLE
    _MPS_AVAILABLE = False  # TODO(lite): revert this once MPS utils have moved
    return [0] if _MPS_AVAILABLE else []


def _get_all_available_cuda_gpus() -> List[int]:
    """
    Returns:
         A list of all available CUDA GPUs
    """
    return list(range(num_cuda_devices()))


def _check_unique(device_ids: List[int]) -> None:
    """Checks that the device_ids are unique.

    Args:
        device_ids: List of ints corresponding to GPUs indices

    Raises:
        MisconfigurationException:
            If ``device_ids`` of GPUs aren't unique
    """
    if len(device_ids) != len(set(device_ids)):
        raise MisconfigurationException("Device ID's (GPU) must be unique.")


def _check_data_type(device_ids: Any) -> None:
    """Checks that the device_ids argument is one of the following: None, int, string, or sequence of integers.

    Args:
        device_ids: gpus/tpu_cores parameter as passed to the Trainer

    Raises:
        MisconfigurationException:
            If ``device_ids`` of GPU/TPUs aren't ``int``, ``str``, sequence of ``int`` or ``None``
    """
    msg = "Device IDs (GPU/TPU) must be an int, a string, a sequence of ints or None, but you passed"

    if device_ids is None:
        return
    elif isinstance(device_ids, (MutableSequence, tuple)):
        for id_ in device_ids:
            if type(id_) is not int:
                raise MisconfigurationException(f"{msg} a sequence of {type(id_).__name__}.")
    elif type(device_ids) not in (int, str):
        raise MisconfigurationException(f"{msg} {type(device_ids).__name__}.")


def _tpu_cores_valid(tpu_cores: Any) -> bool:
    # allow 1 or 8 cores
    if tpu_cores in (1, 8, None):
        return True

    # allow picking 1 of 8 indexes
    if isinstance(tpu_cores, (list, tuple, set)):
        has_1_tpu_idx = len(tpu_cores) == 1
        is_valid_tpu_idx = 1 <= list(tpu_cores)[0] <= 8

        is_valid_tpu_core_choice = has_1_tpu_idx and is_valid_tpu_idx
        return is_valid_tpu_core_choice

    return False


def _parse_tpu_cores_str(tpu_cores: str) -> Union[int, List[int]]:
    if tpu_cores in ("1", "8"):
        return int(tpu_cores)
    return [int(x.strip()) for x in tpu_cores.split(",") if len(x) > 0]


def num_cuda_devices() -> int:
    """Returns the number of GPUs available.

    Unlike :func:`torch.cuda.device_count`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.
    """
    if "fork" not in torch.multiprocessing.get_all_start_methods() or _is_forking_disabled():
        return torch.cuda.device_count()
    with multiprocessing.get_context("fork").Pool(1) as pool:
        return pool.apply(torch.cuda.device_count)


def is_cuda_available() -> bool:
    """Returns a bool indicating if CUDA is currently available.

    Unlike :func:`torch.cuda.is_available`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.
    """
    if "fork" not in torch.multiprocessing.get_all_start_methods() or _is_forking_disabled():
        return torch.cuda.is_available()
    with multiprocessing.get_context("fork").Pool(1) as pool:
        return pool.apply(torch.cuda.is_available)
