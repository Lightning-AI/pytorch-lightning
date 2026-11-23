"""Measure the phases of CPU worker startup independently."""

import multiprocessing as mp
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory


def report(label, start):
    print(f"{label}: {time.perf_counter() - start:.3f}s", flush=True)


def worker(rank, case, rendezvous, started, tensor=None):
    report(f"{case} rank {rank} reached target", started)
    if case == "python":
        return
    start = time.perf_counter()
    import torch

    report(f"{case} rank {rank} imported torch", start)
    if tensor is not None:
        assert tensor.item() == 1
    if case not in ("tcp", "file"):
        return

    dist = torch.distributed
    timeout = timedelta(seconds=45)
    start = time.perf_counter()
    if case == "file":
        store = dist.FileStore(rendezvous, 2)
    else:
        store = dist.TCPStore(
            "127.0.0.1", rendezvous, 2, rank == 0, timeout,
            use_libuv=os.environ["USE_LIBUV"] == "1",
        )
    report(f"{case} rank {rank} created store", start)
    start = time.perf_counter()
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2, timeout=timeout)
    report(f"{case} rank {rank} joined process group", start)
    try:
        start = time.perf_counter()
        value = torch.ones(1)
        dist.all_reduce(value)
        assert value.item() == 2
        report(f"{case} rank {rank} reduced tensor", start)
    finally:
        start = time.perf_counter()
        dist.destroy_process_group()
        report(f"{case} rank {rank} destroyed process group", start)


def measure(case, tensor=None, rendezvous=None):
    ctx = mp.get_context("spawn")
    if case == "tcp":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            rendezvous = sock.getsockname()[1]

    started = time.perf_counter()
    processes = [
        ctx.Process(target=worker, args=(rank, case, rendezvous, started, tensor))
        for rank in range(2)
    ]
    try:
        for rank, process in enumerate(processes):
            start = time.perf_counter()
            process.start()
            report(f"{case} rank {rank} parent start returned", start)
        for process in processes:
            process.join(60)
            if process.exitcode != 0:
                raise RuntimeError(f"{case} worker ended with exit code {process.exitcode}")
        report(f"{case} all workers exited", started)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def main():
    print(f"{platform.platform()} / Python {sys.version.split()[0]}", flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], check=True)
    report("fresh interpreter plus Torch import", start)

    import torch

    measure("python")
    measure("torch")
    measure("tensor", tensor=torch.ones(1))
    for interface in ("lo0", "en0"):
        os.environ["GLOO_SOCKET_IFNAME"] = interface
        for case, libuv in (("tcp", "1"), ("file", "1"), ("tcp", "0")):
            os.environ["USE_LIBUV"] = libuv
            print(f"Gloo interface={interface} store={case} USE_LIBUV={libuv}", flush=True)
            with TemporaryDirectory() as directory:
                try:
                    measure(case, rendezvous=str(Path(directory) / "store"))
                except RuntimeError as error:
                    print(f"Probe failed: {error}", flush=True)



if __name__ == "__main__":
    main()
