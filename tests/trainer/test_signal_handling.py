import os
import platform
import signal
import time

import pytest
import torch

from pytorch_lightning import Trainer, Callback
from tests.base import EvalModelTemplate


from torch.multiprocessing import Process

from tests.base.develop_utils import pl_multi_process_test


class KillCallback(Callback):

    def __init__(self, signal):
        self._signal = signal

    # simulate a termination signal
    def on_batch_end(self, trainer, pl_module):
        # send the signal after the first batch
        assert trainer.global_step == 0, "did not interrupt training after first batch"
        pid = os.getpid()
        os.kill(pid, self._signal)

    def on_train_end(self, trainer, pl_module):
        print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEENNNNNNNNNNNNNNNNNNNNNNNNNNND')
        assert trainer._teardown_already_run


    def on_keyboard_interrupt(self, trainer, pl_module):
        print('interrupted')
        # assert trainer.global_rank == 0
        assert trainer.interrupted


def trigger_fatal_signal(trainer):
    model = EvalModelTemplate()
    trainer.fit(model)


def get_available_signal_codes():
    codes = [signal.SIGINT]
    if platform.system() != "Windows":
        codes += [signal.SIGTERM, signal.SIGSEGV]
    codes = [pytest.param(c) for c in codes]
    return codes


@pytest.mark.parametrize(['signal_code'], get_available_signal_codes())
def test_graceful_training_shutdown(signal_code):
    trainer = Trainer(
        max_epochs=100,
        distributed_backend='ddp_cpu',
        callbacks=[KillCallback(signal_code)],
        num_processes=4
    )
    model = EvalModelTemplate()
    trainer.fit(model)


# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Test requires multiple GPUs.')
# @pytest.mark.parametrize(['signal_code'], get_available_signal_codes())
# def test_graceful_training_shutdown_gpu(signal_code):
#     trainer = Trainer(
#         max_epochs=100,
#         distributed_backend='ddp',
#         callbacks=[KillCallback(signal_code)],
#         gpus=2
#     )
#     model = EvalModelTemplate()
#     trainer.fit(model)
