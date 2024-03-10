import os
import logging
import shutil
import subprocess
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
import torch.distributed as dist
import torch.multiprocessing as mp

import torch

system_check_dir = Path("./system_check")


def main():
    setup_logging()
    
    # if not dist.is_available():
    #     raise RuntimeError("Requires PyTorch distributed to be available.")

    if num_cuda_devices() == 0:
        print0("Warning: Skipping system check because no GPUs were detected.")

    if num_cuda_devices() == 1:
        describe_nvidia_smi()
        pass

    if num_cuda_devices() > 1:
        describe_nvidia_smi()
        describe_gpu_connectivity()
        mp.spawn(_check_cuda_distributed, nprocs=num_cuda_devices())


def _check_cuda_distributed(local_rank: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(num_cuda_devices())
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_FILE"] = str(system_check_dir / f"nccl-rank-{local_rank}.txt")

    dist.init_process_group(
        backend="nccl",
        world_size=num_cuda_devices(),
        rank=local_rank,
        timeout=timedelta(seconds=30),
    )

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    dist.barrier()
    payload = torch.rand(100, 100, device=device)
    dist.all_reduce(payload)


def setup_logging() -> None:
    if system_check_dir.is_dir():
        shutil.rmtree(system_check_dir)
    system_check_dir.mkdir()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(system_check_dir / "logs.txt"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


@lru_cache()
def rank() -> int:
    import torch.distributed as dist

    return dist.get_rank()


@lru_cache()
def world_size() -> int:
    import torch.distributed as dist

    return dist.get_world_size()


def print0(*args: Any, **kwargs: Any) -> None:
    if rank() == 0:
        print(*args, **kwargs)


@lru_cache()
def num_cuda_devices() -> int:
    import torch

    return torch.cuda.device_count()


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def collect_nvidia_smi_topo() -> str:
    return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout


def collect_nvidia_smi() -> str:
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout


def describe_nvidia_smi():
    logger = logging.getLogger()
    logger.info(
        "Below is the output of `nvidia-smi`. It shows information about the GPUs that are installed on this machine,"
        " the driver version, and the maximum supported CUDA version it can run.\n"
    )
    logger.info(collect_nvidia_smi())


def describe_gpu_connectivity():
    logger = logging.getLogger()
    logger.info(
        "The matrix below shows how the GPUs in this machine are connected."
        " NVLink (NV) is the fastest connection, and is only available on high-end systems like V100 or A100.\n"
    )
    logger.info(collect_nvidia_smi_topo())


if __name__ == '__main__':
    main()
