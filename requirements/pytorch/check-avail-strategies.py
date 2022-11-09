import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)

    if "bagua" in args:
        import bagua  # noqa: F401
    if "deepspeed" in args:
        import deepspeed  # noqa: F401
    if "fairscale" in args:
        import fairscale  # noqa: F401
    if "horovod" in args:
        import horovod.torch

        # returns an error code
        assert horovod.torch.nccl_built()
