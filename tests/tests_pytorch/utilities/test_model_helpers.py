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
import inspect
import logging

import pytest
import torch.nn
from lightning.pytorch import LightningDataModule
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.utilities.model_helpers import _ModuleMode, _restricted_classmethod, is_overridden
from lightning_utilities import module_available


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


class RestrictedClass:
    @_restricted_classmethod
    def restricted_cmethod(cls):
        # Can only be called on the class type
        pass

    @classmethod
    def cmethod(cls):
        # Can be called on instance or class type
        pass


def test_restricted_classmethod():
    restricted_method = RestrictedClass().restricted_cmethod  # no exception when getting restricted method

    with pytest.raises(TypeError, match="cannot be called on an instance"):
        restricted_method()

    _ = inspect.getmembers(RestrictedClass())  # no exception on inspecting instance


def test_module_mode():
    class ChildChildModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.dropout = torch.nn.Dropout()

    class ChildModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = ChildChildModule()
            self.dropout = torch.nn.Dropout()

    class RootModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child1 = ChildModule()
            self.child2 = ChildModule()
            self.norm = torch.nn.BatchNorm1d(2)

    # Model with all submodules in the same mode
    model = RootModule()
    model.train()
    mode = _ModuleMode()
    mode.capture(model)
    model.eval()
    assert all(not m.training for m in model.modules())
    mode.restore(model)
    assert model.training
    assert all(m.training for m in model.modules())
    model.eval()

    mode = _ModuleMode()
    mode.capture(model)
    model.eval()
    assert all(not m.training for m in model.modules())
    mode.restore(model)
    assert all(not m.training for m in model.modules())
    model.train()

    # Model with submodules in different modes
    model.norm.eval()
    model.child1.eval()
    model.child2.train()
    model.child2.child.eval()
    model.child2.child.layer.train()

    mode = _ModuleMode()
    mode.capture(model)
    model.eval()
    assert all(not m.training for m in model.modules())
    mode.restore(model)
    assert model.training
    assert not model.norm.training
    assert all(not m.training for m in model.child1.modules())
    assert model.child2.training
    assert model.child2.dropout.training
    assert not model.child2.child.training
    assert model.child2.child.layer.training
    assert not model.child2.child.dropout.training


def test_module_mode_restore_missing_module():
    """Test that restoring still works if the module drops a layer after it was captured."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child1 = torch.nn.Linear(2, 2)
            self.child2 = torch.nn.Linear(2, 2)

    model = Model()
    mode = _ModuleMode()
    mode.capture(model)
    model.child1.eval()
    del model.child2
    assert not hasattr(model, "child2")
    mode.restore(model)
    assert model.child1.training


def test_module_mode_restore_new_module(caplog):
    """Test that restoring ignores newly added submodules after the module was captured."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(2, 2)

    model = Model()
    mode = _ModuleMode()
    mode.capture(model)
    model.child.eval()
    model.new_child = torch.nn.Linear(2, 2)
    with caplog.at_level(logging.DEBUG, logger="lightning.pytorch.utilities.model_helpers"):
        mode.restore(model)
    assert "Restoring training mode on module 'new_child' not possible" in caplog.text


def test_module_mode_clear():
    class Model1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child1 = torch.nn.Linear(2, 2)

    class Model2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child2 = torch.nn.Linear(2, 2)

    model1 = Model1()
    model2 = Model2()
    mode = _ModuleMode()
    mode.capture(model1)
    assert mode.mode == {"": True, "child1": True}
    mode.capture(model2)
    assert mode.mode == {"": True, "child2": True}  # child1 is not included anymore
