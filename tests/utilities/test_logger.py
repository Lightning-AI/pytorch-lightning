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
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.logger import (
    _add_prefix,
    _convert_params,
    _flatten_dict,
    _name,
    _sanitize_callable_params,
    _sanitize_params,
    _version,
)


def test_convert_params():
    """Test conversion of params to a dict."""

    # Test normal dict, make sure it is unchanged
    params = {"foo": "bar", 1: 23}
    assert type(params) == dict
    params = _convert_params(params)
    assert type(params) == dict
    assert params["foo"] == "bar"
    assert params[1] == 23

    # Test None conversion
    params = None
    assert type(params) != dict
    params = _convert_params(params)
    assert type(params) == dict
    assert params == {}

    # Test conversion of argparse Namespace
    opt = "--max_epochs 1".split(" ")
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parent_parser=parser)
    params = parser.parse_args(opt)

    assert type(params) == Namespace
    params = _convert_params(params)
    assert type(params) == dict
    assert params["gpus"] is None


def test_flatten_dict():
    """Validate flatten_dict can handle nested dictionaries and argparse Namespace."""

    # Test basic dict flattening with custom delimiter
    params = {"a": {"b": "c"}}
    params = _flatten_dict(params, "--")

    assert "a" not in params
    assert params["a--b"] == "c"

    # Test complex nested dict flattening
    params = {"a": {5: {"foo": "bar"}}, "b": 6, "c": {7: [1, 2, 3, 4], 8: "foo", 9: {10: "bar"}}}
    params = _flatten_dict(params)

    assert "a" not in params
    assert params["a/5/foo"] == "bar"
    assert params["b"] == 6
    assert params["c/7"] == [1, 2, 3, 4]
    assert params["c/8"] == "foo"
    assert params["c/9/10"] == "bar"

    # Test flattening of argparse Namespace
    opt = "--max_epochs 1".split(" ")
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parent_parser=parser)
    params = parser.parse_args(opt)
    wrapping_dict = {"params": params}
    params = _flatten_dict(wrapping_dict)

    assert type(params) == dict
    assert params["params/logger"] is True
    assert params["params/gpus"] == "None"
    assert "logger" not in params
    assert "gpus" not in params


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


def test_sanitize_params():
    """Verify sanitize params converts various types to loggable strings."""

    params = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "list": [1, 2, 3],
        "np_bool": np.bool_(False),
        "np_int": np.int_(5),
        "np_double": np.double(3.14159),
        "namespace": Namespace(foo=3),
        "layer": torch.nn.BatchNorm1d,
        "tensor": torch.ones(3),
    }
    params = _sanitize_params(params)

    assert params["float"] == 0.3
    assert params["int"] == 1
    assert params["string"] == "abc"
    assert params["bool"] is True
    assert params["list"] == "[1, 2, 3]"
    assert params["np_bool"] is False
    assert params["np_int"] == 5
    assert params["np_double"] == 3.14159
    assert params["namespace"] == "Namespace(foo=3)"
    assert params["layer"] == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"
    assert torch.equal(params["tensor"], torch.ones(3))


def test_add_prefix():
    """Verify add_prefix modifies the dict keys correctly."""

    metrics = {"metric1": 1, "metric2": 2}
    metrics = _add_prefix(metrics, "prefix", "-")

    assert "prefix-metric1" in metrics
    assert "prefix-metric2" in metrics
    assert "metric1" not in metrics
    assert "metric2" not in metrics

    metrics = _add_prefix(metrics, "prefix2", "_")

    assert "prefix2_prefix-metric1" in metrics
    assert "prefix2_prefix-metric2" in metrics
    assert "prefix-metric1" not in metrics
    assert "prefix-metric2" not in metrics
    assert metrics["prefix2_prefix-metric1"] == 1
    assert metrics["prefix2_prefix-metric2"] == 2


def test_name(tmpdir):
    """Verify names of loggers are concatenated properly."""
    logger1 = CSVLogger(tmpdir, name="foo")
    logger2 = CSVLogger(tmpdir, name="bar")
    logger3 = CSVLogger(tmpdir, name="foo")
    logger4 = CSVLogger(tmpdir, name="baz")
    loggers = [logger1, logger2, logger3, logger4]
    name = _name([])
    assert name == ""
    name = _name([logger3])
    assert name == "foo"
    name = _name(loggers)
    assert name == "foo_bar_baz"
    name = _name(loggers, "-")
    assert name == "foo-bar-baz"


def test_version(tmpdir):
    """Verify versions of loggers are concatenated properly."""
    logger1 = CSVLogger(tmpdir, version=0)
    logger2 = CSVLogger(tmpdir, version=2)
    logger3 = CSVLogger(tmpdir, version=1)
    logger4 = CSVLogger(tmpdir, version=0)
    loggers = [logger1, logger2, logger3, logger4]
    version = _version([])
    assert version == ""
    version = _version([logger3])
    assert version == 1
    version = _version(loggers)
    assert version == "0_2_1"
    version = _version(loggers, "-")
    assert version == "0-2-1"
