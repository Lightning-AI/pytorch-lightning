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

SYSTEM_CHECK_DIR = Path("./system_check")


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
        
        context = mp.spawn(
            _check_cuda_distributed,
            nprocs=num_cuda_devices,
            args=(num_cuda_devices,),
            join=False,
        )

        start = time.time()
        joined = False
        while not joined and (time.time() - start < timeout):
            joined = context.join(timeout=5)
            time.sleep(1)

        if not joined:
            for pid in context.pids():
                _kill_process(pid)
            print("not successful")  # TODO

        # TODO: relative dir
        relative_dir = SYSTEM_CHECK_DIR.relative_to(Path.cwd())
        print(f"Find detailed logs at {relative_dir}")


def _check_cuda_distributed(local_rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_FILE"] = str(SYSTEM_CHECK_DIR / f"nccl-rank-{local_rank}.txt")

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

    # TODO: remove
    # if local_rank > 0:
    #     return
    
    _print0(
        "Synchronizing GPUs. If the program hangs for more than 30 seconds, there is a problem with your"
        " multi-GPU setup."
    )
    dist.barrier()

    payload = torch.rand(100, 100, device=device)
    _print0("Running all-reduce test ... ", end="")
    dist.all_reduce(payload)
    _print0("Done.")


def _setup_logging() -> None:
    if SYSTEM_CHECK_DIR.is_dir():
        shutil.rmtree(SYSTEM_CHECK_DIR)
    SYSTEM_CHECK_DIR.mkdir()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(SYSTEM_CHECK_DIR / "logs.txt"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def _print0(*args: Any, **kwargs: Any) -> None:
    if int(os.getenv("RANK", 0)) == 0:
        print(*args, **kwargs)


def _collect_nvidia_smi_topo() -> str:
    return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout


def _collect_nvidia_smi() -> str:
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout


def _describe_nvidia_smi() -> None:
    logger = logging.getLogger()
    logger.info(
        "Below is the output of `nvidia-smi`. It shows information about the GPUs that are installed on this machine,"
        " the driver version, and the maximum supported CUDA version it can run.\n"
    )
    logger.info(_collect_nvidia_smi())


def _describe_gpu_connectivity() -> None:
    logger = logging.getLogger()
    logger.info(
        "The matrix below shows how the GPUs in this machine are connected."
        " NVLink (NV) is the fastest connection, and is only available on high-end systems like V100 or A100.\n"
    )
    logger.info(_collect_nvidia_smi_topo())


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
