# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os
import re
import traceback
from contextlib import contextmanager
from typing import Optional, Type

import pytest

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.loggers import TensorBoardLogger
from tests_pytorch import _TEMP_PATH, RANDOM_PORTS


def get_default_logger(save_dir, version=None):
    # set up logger object without actually saving logs
    logger = TensorBoardLogger(save_dir, name="lightning_logs", version=version)
    return logger


def get_data_path(expt_logger, path_dir=None):
    # some calls contain only experiment not complete logger

    # each logger has to have these attributes
    name, version = expt_logger.name, expt_logger.version

    # the other experiments...
    if not path_dir:
        if hasattr(expt_logger, "save_dir") and expt_logger.save_dir:
            path_dir = expt_logger.save_dir
        else:
            path_dir = _TEMP_PATH
    path_expt = os.path.join(path_dir, name, "version_%s" % version)

    # try if the new sub-folder exists, typical case for test-tube
    if not os.path.isdir(path_expt):
        path_expt = path_dir
    return path_expt


def load_model_from_checkpoint(logger, root_weights_dir, module_class=BoringModel):
    trained_model = module_class.load_from_checkpoint(root_weights_dir)
    assert trained_model is not None, "loading model failed"
    return trained_model


def assert_ok_model_acc(trainer, key="test_acc", thr=0.5):
    # this model should get 0.80+ acc
    acc = trainer.callback_metrics[key]
    assert acc > thr, f"Model failed to get expected {thr} accuracy. {key} = {acc}"


def reset_seed(seed=0):
    seed_everything(seed)


def set_random_main_port():
    reset_seed()
    port = RANDOM_PORTS.pop()
    os.environ["MASTER_PORT"] = str(port)


def init_checkpoint_callback(logger):
    checkpoint = ModelCheckpoint(dirpath=logger.save_dir)
    return checkpoint


def pl_multi_process_test(func):
    """Wrapper for running multi-processing tests_pytorch."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        from multiprocessing import Process, Queue

        queue = Queue()

        def inner_f(queue, **kwargs):
            try:
                func(**kwargs)
                queue.put(1)
            except Exception:
                _trace = traceback.format_exc()
                print(_trace)
                # code 17 means RuntimeError: tensorflow/compiler/xla/xla_client/mesh_service.cc:364 :
                # Failed to meet rendezvous 'torch_xla.core.xla_model.save': Socket closed (14)
                if "terminated with exit code 17" in _trace:
                    queue.put(1)
                else:
                    queue.put(-1)

        proc = Process(target=inner_f, args=(queue,), kwargs=kwargs)
        proc.start()
        proc.join()

        result = queue.get()
        assert result == 1, "expected 1, but returned %s" % result

    return wrapper


@contextmanager
def no_warning_call(expected_warning: Type[Warning] = UserWarning, match: Optional[str] = None):
    with pytest.warns(None) as record:
        yield

    if match is None:
        try:
            w = record.pop(expected_warning)
        except AssertionError:
            # no warning raised
            return
    else:
        for w in record.list:
            if w.category is expected_warning and re.compile(match).search(w.message.args[0]):
                break
        else:
            return

    msg = "A warning" if expected_warning is None else f"`{expected_warning.__name__}`"
    raise AssertionError(f"{msg} was raised: {w}")
