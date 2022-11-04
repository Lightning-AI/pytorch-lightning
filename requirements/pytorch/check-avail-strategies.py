import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)

    if not args or "bagua" in args:
        import bagua  # noqa: F401
    if not args or "deepspeed" in args:
        import deepspeed  # noqa: F401
    if not args or "fairscale" in args:
        import fairscale  # noqa: F401
    if not args or "horovod" in args:
        import horovod.torch

        # returns an error code
        assert horovod.torch.nccl_built()
