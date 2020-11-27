import time
from typing import Callable

import torch
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything


def record_ddp_fit_model_stats(trainer, model, use_cuda):
    """
    Helper to calculate wall clock time for fit + max allocated memory.

    Args:
        trainer: The trainer object.
        model: The model to fit.
        use_cuda: Whether to sync CUDA kernels.

    Returns:
        Max Memory if using GPUs, and total wall clock time.
    """
    max_memory = None

    time_start = time.perf_counter()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    trainer.fit(model)

    if use_cuda:
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 2 ** 20

    total_time = time.perf_counter() - time_start

    return max_memory, total_time


def plugin_parity_test(
        model_cls: Callable,
        plugin: DDPPlugin,
        seed: int = 42,
        accelerator: str = 'ddp_spawn',
        gpus: int = 0,
        precision: int = 32,
        max_percent_speed_diff: float = 0.25):
    """
    Ensures that the trained model is identical to the standard DDP implementation.
    Also checks for speed/memory regressions, we should expect always less memory but performance to fluctuate.

    Args:
        model_cls: Model class to use for test.
        plugin: Plugin to parity test.
        seed: Seed for generators. Note that this does not handle the seed for data-loading on multi-process.
        accelerator: Accelerator type for test.
        gpus: Number of GPUS to enable.
        precision: Whether to use AMP or normal FP32 training.
        max_percent_speed_diff: The maximum speed difference compared to normal DDP training.
        This is more a safety net for variability in CI which can vary in speed, not for benchmarking.

    """

    # Train normal DDP
    seed_everything(seed)
    ddp_model = model_cls()
    use_cuda = gpus > 0

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
    )

    max_memory_ddp, ddp_time = record_ddp_fit_model_stats(
        trainer=trainer,
        model=ddp_model,
        use_cuda=use_cuda
    )

    # Reset and train Custom DDP
    seed_everything(seed)
    custom_plugin_model = model_cls()

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
        plugins=[plugin],
    )

    max_memory_custom, custom_model_time = record_ddp_fit_model_stats(
        trainer=trainer,
        model=custom_plugin_model,
        use_cuda=use_cuda
    )

    # Assert model parameters are identical after fit
    for ddp_param, custom_param in zip(ddp_model.parameters(), custom_plugin_model.parameters()):
        assert torch.equal(ddp_param, custom_param), 'Model parameters are different between DDP and Custom plugin'

    # Assert speed parity by ensuring percentage difference between custom/ddp is below threshold
    percent_diff = (custom_model_time - ddp_time) / custom_model_time

    assert percent_diff <= max_percent_speed_diff, \
        f'Custom DDP plugin was too slow compared to DDP, Custom Plugin Time: {custom_model_time}, DDP Time: {ddp_time}'

    if use_cuda:
        # Assert CUDA memory parity
        assert max_memory_custom <= max_memory_ddp, \
            f'Custom plugin used too much memory compared to DDP,' \
            f'Custom Mem: {max_memory_custom}, DDP Mem: {max_memory_ddp}'
