import os
import threading
import time
from signal import SIGINT, SIGTERM

import pytest

from pytorch_lightning import Trainer, Callback
from tests.base import EvalModelTemplate


from torch.multiprocessing import Process, Queue



def trigger_fatal_signal():
    pid = os.getpid()
    model = EvalModelTemplate()

    class KillCallback(Callback):

        def on_batch_end(self, trainer, pl_module):
            os.kill(pid, SIGTERM)

    trainer = Trainer(max_steps=3, distributed_backend='ddp', callbacks=[KillCallback()])
    trainer.fit(model)
    # if trainer._teardown_already_run


# @pytest.mark.parametrize()
def test_graceful_training_shutdown():
    p = Process(target=trigger_fatal_signal)
    p.start()
    p.join(timeout=10)
    assert p.exitcode == SIGINT
    # p.join() # this blocks until the process terminates
    # result = queue.get()
    # print(result)

if __name__ == '__main__':
    test_graceful_training_shutdown()