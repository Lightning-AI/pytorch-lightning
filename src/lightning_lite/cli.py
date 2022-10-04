import os
from argparse import ArgumentParser
import torch.distributed.run as torchrun


def main():
    parser = ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("--accelerator", type=str, default="cpu", choices=("cpu", "cuda", "mps", "tpu", "auto"))
    # TODO: note for some accelerators/strategies, torchrun won't make sense (e.g. dp)
    # TODO: should we include spawn?
    parser.add_argument("--strategy", type=str, default=None, choices=(None, "ddp", "dp", "deepspeed"))
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--main-address", type=str, default="127.0.0.1")
    parser.add_argument("--main-port", type=int, default=29400)
    parser.add_argument("--precision", type=str, default="32", choices=("32", "16", "bf16"))
    args = parser.parse_args()

    os.environ["LT_ACCELERATOR"] = str(args.accelerator)
    if args.strategy:
        os.environ["LT_STRATEGY"] = str(args.strategy)
    os.environ["LT_DEVICES"] = str(args.devices)
    os.environ["LT_NUM_NODES"] = str(args.num_nodes)
    os.environ["LT_PRECISION"] = str(args.precision)

    num_devices = int(args.devices)  # TODO: count them

    torchrun_args = []
    torchrun_args.extend(["--nproc_per_node", str(num_devices)])
    torchrun_args.extend(["--nnodes", str(args.num_nodes)])
    torchrun_args.extend(["--node_rank", str(args.node_rank)])
    torchrun_args.extend(["--master_addr", args.main_address])
    torchrun_args.extend(["--master_port", str(args.main_port)])
    torchrun_args.append(args.script)

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() // num_devices)))

    torchrun.main(torchrun_args)


if __name__ == "__main__":
    main()
