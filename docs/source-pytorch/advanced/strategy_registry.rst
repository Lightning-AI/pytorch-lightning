Strategy Registry
=================

Lightning includes a registry that holds information about Training strategies and allows for the registration of new custom strategies.

The Strategies are assigned strings that identify them, such as "ddp", "deepspeed_stage_2_offload", and so on.
It also returns the optional description and parameters for initialising the Strategy that were defined during registration.


.. code-block:: python

    # Training with the DDP Strategy
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Training with DeepSpeed ZeRO Stage 3 and CPU Offload
    trainer = Trainer(strategy="deepspeed_stage_3_offload", accelerator="gpu", devices=3)

    # Training with the TPU Spawn Strategy with `debug` as True
    trainer = Trainer(strategy="xla_debug", accelerator="tpu", devices=8)


Additionally, you can pass your custom registered training strategies to the ``strategy`` argument.

.. code-block:: python

    from lightning.pytorch.strategies import DDPStrategy, StrategyRegistry, CheckpointIO


    class CustomCheckpointIO(CheckpointIO):
        def save_checkpoint(self, checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
            ...

        def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
            ...


    custom_checkpoint_io = CustomCheckpointIO()

    # Register the DDP Strategy with your custom CheckpointIO plugin
    StrategyRegistry.register(
        "ddp_custom_checkpoint_io",
        DDPStrategy,
        description="DDP Strategy with custom checkpoint io plugin",
        checkpoint_io=custom_checkpoint_io,
    )

    trainer = Trainer(strategy="ddp_custom_checkpoint_io", accelerator="gpu", devices=2)
