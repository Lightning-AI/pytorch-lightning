import os
import signal
import time

import pytest

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

    def on_keyboard_interrupt(self, trainer, pl_module):
        print('interrupted')
        # assert trainer.global_rank == 0
        assert trainer.interrupted


def trigger_fatal_signal(trainer):
    model = EvalModelTemplate()
    trainer.fit(model)


@pytest.mark.parametrize(['signal_code'], [
    pytest.param(signal.SIGINT),
    pytest.param(signal.SIGTERM),
    pytest.param(signal.SIGSEGV),
])
#@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
#@pl_multi_process_test
def test_graceful_training_shutdown(signal_code):
    #signal_code = signal.SIGINT
    trainer = Trainer(max_epochs=100, distributed_backend='ddp_cpu', callbacks=[KillCallback(signal_code)], num_processes=4)
    model = EvalModelTemplate()
    #with pytest.raises(SystemExit):
    trainer.fit(model)
    #assert trainer.global_step == 0
    #assert trainer.batch_idx == 0
    # p = Process(target=trigger_fatal_signal, args=(trainer, ))
    # start = time.time()
    # timeout = 30  # seconds
    # p.start()
    # # wait until Trainer gets killed
    # while p.is_alive():
    #     assert time.time() - start < timeout

    # assert p.exitcode == signal_code
    # assert trainer.global_step == 1
    # assert trainer.interrupted

#@test_graceful_training_shutdown(signal.SIGINT)