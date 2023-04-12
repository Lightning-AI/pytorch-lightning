if __name__ == "__main__":
    import deepspeed  # noqa: F401
    import lightning_bagua  # noqa: F401

    # colossalai does not support torch 2.0
    # import lightning_colossalai
