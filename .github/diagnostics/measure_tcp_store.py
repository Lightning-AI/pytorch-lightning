"""Measure Python socket operations and each PyTorch rendezvous mode."""

import multiprocessing
import socket
import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.distributed as dist


def report(label, started):
    print(f"{label}: {time.perf_counter() - started:.3f}s", flush=True)


def worker(rank, directory, port, mode):
    timeout = timedelta(seconds=50)
    started = time.perf_counter()
    if mode == "file":
        store = dist.FileStore(str(Path(directory) / "store"), 2)
    else:
        store = dist.TCPStore(
            "127.0.0.1", port, 2, rank == 0, timeout, wait_for_workers=(mode == "tcp-wait")
        )
    report(f"{mode} rank {rank} store construction", started)

    started = time.perf_counter()
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2, timeout=timeout)
    report(f"{mode} rank {rank} Gloo initialization", started)
    try:
        dist.barrier()
    finally:
        dist.destroy_process_group()


def measure(mode, directory):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, directory, port, mode)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(65)
        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(f"{mode} rank process ended with exit code {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def main():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        endpoint = listener.getsockname()
        started = time.perf_counter()
        with socket.create_connection(endpoint, timeout=10), listener.accept()[0]:
            report("raw loopback connection", started)

    address_flags = socket.AI_V4MAPPED | socket.AI_ALL | socket.AI_NUMERICSERV
    started = time.perf_counter()
    socket.getaddrinfo("127.0.0.1", 29500, socket.AF_INET6, socket.SOCK_STREAM, flags=address_flags)
    report("IPv4-mapped getaddrinfo", started)
    for address in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "::"):
        started = time.perf_counter()
        result = socket.getnameinfo((address, 29500), socket.NI_NUMERICSERV)
        report(f"getnameinfo({address}) -> {result[0]}", started)

    for mode in ("file", "tcp-no-wait", "tcp-wait"):
        print(f"::group::{mode}", flush=True)
        try:
            with TemporaryDirectory() as directory:
                measure(mode, directory)
        except RuntimeError as error:
            print(f"::warning::{error}", flush=True)
        finally:
            print("::endgroup::", flush=True)


if __name__ == "__main__":
    main()
