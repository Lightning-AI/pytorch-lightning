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
from contextlib import nullcontext
from typing import Dict, Generic, Iterator, Mapping, TypeVar

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.automatic import ClosureResult
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_closure_result_deepcopy():
    closure_loss = torch.tensor(123.45)
    result = ClosureResult(closure_loss)

    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    # the `loss` is cloned so the storage is different
    assert closure_loss.data_ptr() != result.loss.data_ptr()

    copy = result.asdict()
    assert result.loss == copy["loss"]
    assert copy.keys() == {"loss"}

    # no copy
    assert id(result.loss) == id(copy["loss"])
    assert result.loss.data_ptr() == copy["loss"].data_ptr()


def test_closure_result_apply_accumulation():
    closure_loss = torch.tensor(25.0)
    result = ClosureResult.from_training_step_output(closure_loss, 5)
    assert result.loss == 5


T = TypeVar("T")


class OutputMapping(Generic[T], Mapping[str, T]):
    def __init__(self, d: Dict[str, T]) -> None:
        self.d: Dict[str, T] = d

    def __iter__(self) -> Iterator[str]:
        return iter(self.d)

    def __len__(self) -> int:
        return len(self.d)

    def __getitem__(self, key: str) -> T:
        return self.d[key]


@pytest.mark.parametrize(
    "case",
    [
        (5.0, "must return a Tensor, a dict, or None"),
        ({"a": 5}, "the 'loss' key needs to be present"),
        (OutputMapping({"a": 5}), "the 'loss' key needs to be present"),
    ],
)
def test_warning_invalid_trainstep_output(tmp_path, case):
    output, match = case

    class InvalidTrainStepModel(BoringModel):
        def training_step(self, batch, batch_idx):
            return output

    model = InvalidTrainStepModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)

    with pytest.raises(MisconfigurationException, match=match):
        trainer.fit(model)


@pytest.mark.parametrize("world_size", [1, 2])
def test_skip_training_step_not_allowed(world_size, tmp_path):
    """Test that skipping the training_step in distributed training is not allowed."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            return None

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        barebones=True,
    )
    trainer.strategy.world_size = world_size  # mock world size without launching processes
    error_context = (
        pytest.raises(RuntimeError, match="Skipping the `training_step` .* is not supported")
        if world_size > 1
        else nullcontext()
    )
    with error_context:
        trainer.fit(model)
