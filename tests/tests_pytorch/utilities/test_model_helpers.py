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
from lightning_utilities import module_available

from lightning.pytorch import LightningDataModule
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.utilities.model_helpers import is_overridden


def test_is_overridden():
    # edge cases
    assert not is_overridden("whatever", None)
    with pytest.raises(ValueError, match="Expected a parent"):
        is_overridden("whatever", object())
    model = BoringModel()
    assert not is_overridden("whatever", model)
    assert not is_overridden("whatever", model, parent=LightningDataModule)
    # normal usage
    assert is_overridden("training_step", model)
    datamodule = BoringDataModule()
    assert is_overridden("train_dataloader", datamodule)


@pytest.mark.skipif(
    not module_available("lightning") or not module_available("pytorch_lightning"),
    reason="This test is ONLY relevant for the UNIFIED package",
)
def test_mixed_imports_unified():
    from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized as new_unwrap
    from lightning.pytorch.utilities.model_helpers import is_overridden as new_is_overridden
    from pytorch_lightning.callbacks import EarlyStopping as OldEarlyStopping
    from pytorch_lightning.demos.boring_classes import BoringModel as OldBoringModel

    model = OldBoringModel()
    with pytest.raises(TypeError, match=r"`pytorch_lightning` object \(BoringModel\) to a `lightning.pytorch`"):
        new_unwrap(model)

    with pytest.raises(TypeError, match=r"`pytorch_lightning` object \(EarlyStopping\) to a `lightning.pytorch`"):
        new_is_overridden("on_fit_start", OldEarlyStopping("foo"))
