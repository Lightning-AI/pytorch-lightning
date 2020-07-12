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
        pid = os.getpid()
        os.kill(pid, self._signal)
        # if trainer.global_step == 2:
        #     raise KeyboardInterrupt

    def on_train_end(self, trainer, pl_module):
        print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEENNNNNNNNNNNNNNNNNNNNNNNNNNND')

    def on_keyboard_interrupt(self, trainer, pl_module):
        print('interrupted')
        assert trainer.interrupted


def trigger_fatal_signal(trainer):
    model = EvalModelTemplate()
    trainer.fit(model)


# @pytest.mark.parametrize(['signal_code'], [
#     pytest.param(),
#     # pytest.param(signal.SIGTERM),
#     # pytest.param(signal.SIGSEGV),
# ])
@pl_multi_process_test
def test_graceful_training_shutdown():
    signal_code = signal.SIGINT
    trainer = Trainer(max_epochs=100, distributed_backend='ddp', callbacks=[KillCallback(signal_code)])
    model = EvalModelTemplate()
    #with pytest.raises(KeyboardInterrupt):
    result = trainer.fit(model)
    assert result
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