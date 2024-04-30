# Copyright The Lightning AI team.
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

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import TensorBoardLogger


def get_default_logger(save_dir, version=None):
    # set up logger object without actually saving logs
    return TensorBoardLogger(save_dir, name="lightning_logs", version=version)


def get_data_path(expt_logger, path_dir):
    # some calls contain only experiment not complete logger

    # each logger has to have these attributes
    name, version = expt_logger.name, expt_logger.version

    # the other experiments...
    path_expt = os.path.join(path_dir, name, "version_%s" % version)

    # try if the new sub-folder exists, typical case for test-tube
    if not os.path.isdir(path_expt):
        path_expt = path_dir
    return path_expt


def load_model_from_checkpoint(root_weights_dir, module_class=BoringModel):
    trained_model = module_class.load_from_checkpoint(root_weights_dir)
    assert trained_model is not None, "loading model failed"
    return trained_model


def assert_ok_model_acc(trainer, key="test_acc", thr=0.5):
    # this model should get 0.80+ acc
    acc = trainer.callback_metrics[key]
    assert acc > thr, f"Model failed to get expected {thr} accuracy. {key} = {acc}"


def init_checkpoint_callback(logger):
    return ModelCheckpoint(dirpath=logger.save_dir)


def getattr_recursive(obj, attr):
    return functools.reduce(getattr, [obj] + attr.split("."))
