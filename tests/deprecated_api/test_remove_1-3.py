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
"""Test deprecated functionality which will be removed in vX.Y.Z"""
from argparse import ArgumentParser
from unittest import mock

import pytest
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profiler.profilers import PassThroughProfiler, SimpleProfiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_tbd_remove_in_v1_3_0(tmpdir):
    with pytest.deprecated_call(match='will no longer be supported in v1.3'):
        callback = ModelCheckpoint()
        Trainer(checkpoint_callback=callback, callbacks=[], default_root_dir=tmpdir)

    # Deprecate prefix
    with pytest.deprecated_call(match='will be removed in v1.3'):
        ModelCheckpoint(prefix='temp')

    # Deprecate auto mode
    with pytest.deprecated_call(match='will be removed in v1.3'):
        ModelCheckpoint(mode='auto')

    with pytest.deprecated_call(match='will be removed in v1.3'):
        EarlyStopping(mode='auto')

    with pytest.deprecated_call(match="The setter for self.hparams in LightningModule is deprecated"):
        class DeprecatedHparamsModel(LightningModule):
            def __init__(self, hparams):
                super().__init__()
                self.hparams = hparams
        DeprecatedHparamsModel({})


def test_tbd_remove_in_v1_3_0_metrics():
    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import to_onehot
        to_onehot(torch.tensor([1, 2, 3]))

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import to_categorical
        to_categorical(torch.tensor([[0.2, 0.5], [0.9, 0.1]]))

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import get_num_classes
        get_num_classes(pred=torch.tensor([0, 1]), target=torch.tensor([1, 1]))

    x_binary = torch.tensor([0, 1, 2, 3])
    y_binary = torch.tensor([0, 1, 2, 3])

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import roc
        roc(pred=x_binary, target=y_binary)

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import _roc
        _roc(pred=x_binary, target=y_binary)

    x_multy = torch.tensor([[0.85, 0.05, 0.05, 0.05],
                            [0.05, 0.85, 0.05, 0.05],
                            [0.05, 0.05, 0.85, 0.05],
                            [0.05, 0.05, 0.05, 0.85]])
    y_multy = torch.tensor([0, 1, 3, 2])

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import multiclass_roc
        multiclass_roc(pred=x_multy, target=y_multy)

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import average_precision
        average_precision(pred=x_binary, target=y_binary)

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import precision_recall_curve
        precision_recall_curve(pred=x_binary, target=y_binary)

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.classification import multiclass_precision_recall_curve
        multiclass_precision_recall_curve(pred=x_multy, target=y_multy)

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.reduction import reduce
        reduce(torch.tensor([0, 1, 1, 0]), 'sum')

    with pytest.deprecated_call(match='will be removed in v1.3'):
        from pytorch_lightning.metrics.functional.reduction import class_reduce
        class_reduce(torch.randint(1, 10, (50,)).float(),
                     torch.randint(10, 20, (50,)).float(),
                     torch.randint(1, 100, (50,)).float())

# TODO: remove bool from Trainer.profiler param in v1.3.0, update profiler_connector.py
@pytest.mark.parametrize(['profiler', 'expected'], [
    (True, SimpleProfiler),
    (False, PassThroughProfiler),
])
def test_trainer_profiler_remove_in_v1_3_0(profiler, expected):
    # remove bool from Trainer.profiler param in v1.3.0, update profiler_connector.py
    with pytest.deprecated_call(match='will be removed in v1.3'):
        trainer = Trainer(profiler=profiler)
        assert isinstance(trainer.profiler, expected)


@pytest.mark.parametrize(
    ['cli_args', 'expected_parsed_arg', 'expected_profiler'],
    [
        ('--profiler', True, SimpleProfiler),
        ('--profiler True', True, SimpleProfiler),
        ('--profiler False', False, PassThroughProfiler),
    ],
)
def test_trainer_cli_profiler_remove_in_v1_3_0(cli_args, expected_parsed_arg, expected_profiler):
    cli_args = cli_args.split(' ')
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = Trainer.add_argparse_args(parent_parser=parser)
        args = Trainer.parse_argparser(parser)

    assert getattr(args, "profiler") == expected_parsed_arg
    trainer = Trainer.from_argparse_args(args)
    assert isinstance(trainer.profiler, expected_profiler)
