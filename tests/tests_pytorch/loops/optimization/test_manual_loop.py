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
import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.manual import ManualResult
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_manual_result():
    training_step_output = {"loss": torch.tensor(25.0, requires_grad=True), "something": "jiraffe"}
    result = ManualResult.from_training_step_output(training_step_output)
    asdict = result.asdict()
    assert not asdict["loss"].requires_grad
    assert asdict["loss"] == 25
    assert result.extra == asdict


def test_warning_invalid_trainstep_output(tmp_path):
    class InvalidTrainStepModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            return 5

    model = InvalidTrainStepModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)

    with pytest.raises(MisconfigurationException, match="return a Tensor or have no return"):
        trainer.fit(model)


def test_manual_result_general_mapping_and_non_tensor_loss():
    from collections.abc import Mapping

    class CustomMapping(Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    mapping_with_tensor = CustomMapping({"loss": torch.tensor(10.0, requires_grad=True), "meta": "ok"})
    result = ManualResult.from_training_step_output(mapping_with_tensor)
    assert isinstance(result.asdict(), dict)
    assert not result.asdict()["loss"].requires_grad
    assert result.asdict()["loss"] == 10.0
    assert result.asdict()["meta"] == "ok"

    mapping_with_none_loss = CustomMapping({"loss": None, "meta": "no_loss"})
    result_none = ManualResult.from_training_step_output(mapping_with_none_loss)
    assert result_none.asdict() == {"loss": None, "meta": "no_loss"}


def test_manual_optimization_resets_optimizer_hooks_on_exception(tmp_path):
    from collections import OrderedDict
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from lightning.pytorch.core.optimizer import do_nothing_closure
    from lightning.pytorch.loops.optimization.manual import _ManualOptimization

    dummy_opt = SimpleNamespace(
        _on_before_step=do_nothing_closure,
        _on_after_step=do_nothing_closure,
    )
    trainer = MagicMock()
    trainer.strategy._lightning_optimizers = [dummy_opt]

    loop = _ManualOptimization(trainer)
    loop.advance = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        loop.run(OrderedDict())

    assert dummy_opt._on_before_step is do_nothing_closure
    assert dummy_opt._on_after_step is do_nothing_closure
    assert not loop._restarting
