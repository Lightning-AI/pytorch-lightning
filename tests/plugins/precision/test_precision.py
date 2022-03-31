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
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DoublePrecisionPlugin, PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_default_precision():
    trainer = Trainer()
    assert trainer.precision == 32
    assert isinstance(trainer.precision_plugin, PrecisionPlugin)


@pytest.mark.parametrize(["precision", "precision_cls"], [(32, PrecisionPlugin), (64, DoublePrecisionPlugin)])
def test_precision_being_passed_to_precision_flag_cpu(precision, precision_cls):

    trainer = Trainer(precision=precision)
    assert trainer.precision == precision
    assert isinstance(trainer.precision_plugin, precision_cls)

    trainer = Trainer(precision=precision_cls())
    assert trainer.precision == precision
    assert isinstance(trainer.precision_plugin, precision_cls)


def test_precision_being_passed_to_precision_and_plugins_flag():

    with pytest.raises(MisconfigurationException, match="you can only specify one precision"):
        Trainer(precision=32, plugins=[PrecisionPlugin()])
