import os
import logging
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import time
from lightning_utilities.core.imports import RequirementCache


_psutil_available = RequirementCache("psutil")
_logger = logging.getLogger(__name__)
_system_check_dir = Path("./system_check")


def main(timeout: int = 60) -> None:
    _setup_logging()
    num_cuda_devices = torch.cuda.device_count()
    
    if num_cuda_devices == 0:
        print("Warning: Skipping system check because no GPUs were detected.")

    if num_cuda_devices == 1:
        # TODO
        _describe_nvidia_smi()
        pass

    if num_cuda_devices > 1:
        _describe_nvidia_smi()
        _describe_gpu_connectivity()
        
        success = _check_cuda_distributed(timeout)
        
        if not success:
            print(
                f"The multi-GPU NCCL test did not finish within {timeout} seconds."
                " It looks like there is an issue with your multi-GPU setup."
                " Now trying to run again with `NCCL_P2P_DISABLE=1` set."
            ) 
            os.environ["NCCL_P2P_DISABLE"] = "1"
            success = _check_cuda_distributed(timeout)
            if not success:
                print(f"Disabling peer-to-peer transport did not fix the issue.")
        else:
            print("Multi-GPU test successful.")

        print(f"Find detailed logs at {_system_check_dir.absolute()}")


def _check_cuda_distributed(timeout: int) -> bool:
    if not _psutil_available:
        raise ModuleNotFoundError(str(_psutil_available))

    num_cuda_devices = torch.cuda.device_count()
    context = mp.spawn(
        _run_all_reduce_test,
        nprocs=num_cuda_devices,
        args=(num_cuda_devices,),
        join=False,
    )

    start = time.time()
    success = False
    while not success and (time.time() - start < timeout):
        success = context.join(timeout=5)
        time.sleep(1)

    if not success:
        for pid in context.pids():
            _kill_process(pid)
    return success


def _run_all_reduce_test(local_rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_FILE"] = str(_system_check_dir / f"nccl-rank-{local_rank}.txt")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    _print0("Setting up the process group ... ", end="")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=local_rank,
        # NCCL gets initialized in the first collective call (e.g., barrier below), 
        # which must be successful for this timeout to work.
        timeout=timedelta(seconds=10),
    )
    _print0("done.")

    # # TODO: remove
    # if local_rank > 0:
    #     return
    
    _print0("Synchronizing GPUs ... ", end="")
    dist.barrier()
    _print0("done.")

    payload = torch.rand(100, 100, device=device)
    _print0("Running all-reduce test ... ", end="")
    dist.all_reduce(payload)
    _print0("done.")


def _setup_logging() -> None:
    if _system_check_dir.is_dir():
        shutil.rmtree(_system_check_dir)
    _system_check_dir.mkdir()

    _logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(_system_check_dir / "logs.txt"))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)


def _print0(string: str, **kwargs: Any) -> None:
    if int(os.getenv("RANK", 0)) == 0:
        _logger.info(string, **kwargs)


def _collect_nvidia_smi_topo() -> str:
    return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout


def _collect_nvidia_smi() -> str:
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout


# def _collect_nvidia_driver_version() -> str:
#     result = subprocess.run(
#         [
#             "nvidia-smi", 
#             "--query-gpu=driver_version", 
#             "--id=0",
#             "--format=csv,noheader",
#         ], 
#         capture_output=True, 
#         text=True,
#     )
#     return result.stdout


def _describe_nvidia_smi() -> None:
    _logger.info(
        "Below is the output of `nvidia-smi`. It shows information about the GPUs that are installed on this machine,"
        " the driver version, and the maximum supported CUDA version it can run.\n"
    )
    _logger.info(_collect_nvidia_smi())


def _describe_gpu_connectivity() -> None:
    _logger.info(
        "The matrix below shows how the GPUs in this machine are connected."
        " NVLink (NV) is the fastest connection, and is only available on high-end systems like V100 or A100.\n"
    )
    _logger.info(_collect_nvidia_smi_topo())


def _kill_process(pid: int) -> None:
    import psutil  # TODO

    try:
        process = psutil.Process(pid)
        if process.is_running():
            process.kill()
    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied:
        pass


if __name__ == '__main__':
    main()
