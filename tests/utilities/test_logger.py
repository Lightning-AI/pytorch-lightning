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
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict, _sanitize_callable_params, _add_prefix
from tests.helpers import BoringModel

def test_flatten_dict():
    """Validate flatten_dict can handle nested dictionaries and argparse Namespace"""

    # Test basic dict flattening with custom delimiter
    params = {"a": {"b" : "c"}}
    params = _flatten_dict(params, "--")

    assert("a" not in params)
    assert(params["a--b"] == "c")

    # Test complex nested dict flattening
    params = {"a": {5: {"foo": "bar"}}, "b": 6, "c": {7: [1,2,3,4], 8: "foo", 9: {10: "bar"}}}
    params = _flatten_dict(params)

    assert("a" not in params)
    assert(params["a/5/foo"] == "bar")
    assert(params["b"] == 6)
    assert(params["c/7"] == [1,2,3,4])
    assert(params["c/8"] == "foo")
    assert(params["c/9/10"] == "bar")

    # Test flattening of argparse Namespace
    opt = "--max_epochs 1".split(" ")
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parent_parser=parser)
    params = parser.parse_args(opt)
    wrapping_dict = {"params": params}
    params = _flatten_dict(wrapping_dict)

    assert(type(params) == dict)
    assert(params["params/logger"] == True)
    assert(params["params/gpus"] == "None")
    assert("logger" not in params)
    assert("gpus" not in params)


def test_sanitize_callable_params():
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

def test_add_prefix():
    """Verify add_prefix modifies the dict keys correctly."""

    metrics = {"metric1" : 1, "metric2" : 2}
    metrics = _add_prefix(metrics, "prefix", "-")

    assert("prefix-metric1" in metrics)
    assert("prefix-metric2" in metrics)
    assert("metric1" not in metrics)
    assert("metric2" not in metrics)

    metrics = _add_prefix(metrics, "prefix2", "_")

    assert("prefix2_prefix-metric1" in metrics)
    assert("prefix2_prefix-metric2" in metrics)
    assert("prefix-metric1" not in metrics)
    assert("prefix-metric2" not in metrics)
    assert(metrics["prefix2_prefix-metric1"] == 1)
    assert(metrics["prefix2_prefix-metric2"] == 2)
