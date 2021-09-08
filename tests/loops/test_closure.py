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
import pickle
from copy import deepcopy

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loops.closure import ClosureResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


def test_optimizer_step_no_closure_raises(tmpdir):
    class TestModel(BoringModel):
        def optimizer_step(
            self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None, **_
        ):
            # does not call `optimizer_closure()`
            pass

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def configure_optimizers(self):
            class BrokenSGD(torch.optim.SGD):
                def step(self, closure=None):
                    # forgot to pass the closure
                    return super().step()

            return BrokenSGD(self.layer.parameters(), lr=0.1)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)


def test_closure_result_deepcopy():
    closure_loss = torch.tensor(123.45)
    hiddens = torch.tensor(321.12, requires_grad=True)
    result = ClosureResult(closure_loss, hiddens)
    assert not result.hiddens.requires_grad

    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    # the `loss` is cloned so the storage is different
    assert closure_loss.data_ptr() != result.loss.data_ptr()

    # make sure `__getstate__` is not missing any keys
    assert vars(result).keys() == result.__getstate__().keys()

    copy = deepcopy(result)
    assert result.loss == copy.loss
    assert copy.closure_loss is None
    assert copy.hiddens is None

    assert id(result.loss) != id(copy.loss)
    assert result.loss.data_ptr() != copy.loss.data_ptr()

    assert copy == pickle.loads(pickle.dumps(result))


def test_closure_result_raises():
    with pytest.raises(MisconfigurationException, match="If `hiddens` are returned .* the loss cannot be `None`"):
        ClosureResult(None, "something")


def test_closure_result_apply_accumulation():
    closure_loss = torch.tensor(25.0)
    result = ClosureResult.from_training_step_output(closure_loss, 5)
    assert result.loss == 5


def test_closure_to():
    result = ClosureResult(torch.tensor(1.0), (torch.tensor(2.0), torch.tensor(3.0)), extra={"foo": torch.tensor(4.0)})
    result.to(torch.half)
    assert result.loss.dtype == torch.half
    assert all(t.dtype == torch.half for t in result.hiddens)
    assert result.extra["foo"].dtype == torch.half
