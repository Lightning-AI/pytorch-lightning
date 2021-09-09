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
    result = ClosureResult(closure_loss)

    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    # the `loss` is cloned so the storage is different
    assert closure_loss.data_ptr() != result.loss.data_ptr()

    copy = result.drop_closure_loss()
    assert result.loss == copy.loss
    assert copy.closure_loss is None

    # no copy
    assert id(result.loss) == id(copy.loss)
    assert result.loss.data_ptr() == copy.loss.data_ptr()


def test_closure_result_apply_accumulation():
    closure_loss = torch.tensor(25.0)
    result = ClosureResult.from_training_step_output(closure_loss, 5)
    assert result.loss == 5
