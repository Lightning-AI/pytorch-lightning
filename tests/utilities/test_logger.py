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
import os
import pickle
from argparse import ArgumentParser
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict, _sanitize_callable_params
from tests.helpers import BoringModel


def test_sanitize_callable_params(tmpdir):
    """Callback function are not serializiable.

    Therefore, we get them a chance to return something and if the returned type is not accepted, return None.
    """
    opt = "--max_epochs 1".split(" ")
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parent_parser=parser)
    params = parser.parse_args(opt)

    def return_something():
        return "something"

    params.something = return_something

    def wrapper_something():
        return return_something

    params.wrapper_something_wo_name = lambda: lambda: "1"
    params.wrapper_something = wrapper_something

    params = _convert_params(params)
    params = _flatten_dict(params)
    params = _sanitize_callable_params(params)
    assert params["gpus"] == "None"
    assert params["something"] == "something"
    assert params["wrapper_something"] == "wrapper_something"
    assert params["wrapper_something_wo_name"] == "<lambda>"
