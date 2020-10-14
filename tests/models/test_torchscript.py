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
from distutils.version import LooseVersion

import pytest
import torch

from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule
from tests.base.models import ParityModuleRNN, BasicGAN


@pytest.mark.parametrize("modelclass", [
    EvalModelTemplate,
    ParityModuleRNN,
    BasicGAN,
])
def test_torchscript_input_output(modelclass):
    """ Test that scripted LightningModule forward works. """
    model = modelclass()
    script = model.to_torchscript()
    assert isinstance(script, torch.jit.ScriptModule)
    model.eval()
    model_output = model(model.example_input_array)
    script_output = script(model.example_input_array)
    assert torch.allclose(script_output, model_output)


@pytest.mark.parametrize("modelclass", [
    EvalModelTemplate,
    ParityModuleRNN,
    BasicGAN,
])
def test_torchscript_input_output_trace(modelclass):
    """ Test that traced LightningModule forward works. """
    model = modelclass()
    script = model.to_torchscript(method='trace')
    assert isinstance(script, torch.jit.ScriptModule)
    model.eval()
    model_output = model(model.example_input_array)
    script_output = script(model.example_input_array)
    assert torch.allclose(script_output, model_output)


@pytest.mark.parametrize("device", [
    torch.device("cpu"),
    torch.device("cuda", 0)
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
def test_torchscript_device(device):
    """ Test that scripted module is on the correct device. """
    model = EvalModelTemplate().to(device)
    script = model.to_torchscript()
    assert next(script.parameters()).device == device
    script_output = script(model.example_input_array.to(device))
    assert script_output.device == device


def test_torchscript_retain_training_state():
    """ Test that torchscript export does not alter the training mode of original model. """
    model = EvalModelTemplate()
    model.train(True)
    script = model.to_torchscript()
    assert model.training
    assert not script.training
    model.train(False)
    _ = model.to_torchscript()
    assert not model.training
    assert not script.training


@pytest.mark.parametrize("modelclass", [
    EvalModelTemplate,
    ParityModuleRNN,
    BasicGAN,
])
def test_torchscript_properties(modelclass):
    """ Test that scripted LightningModule has unnecessary methods removed. """
    model = modelclass()
    model.datamodule = TrialMNISTDataModule()
    script = model.to_torchscript()
    assert not hasattr(script, "datamodule")
    assert not hasattr(model, "batch_size") or hasattr(script, "batch_size")
    assert not hasattr(model, "learning_rate") or hasattr(script, "learning_rate")

    if LooseVersion(torch.__version__) >= LooseVersion("1.4.0"):
        # only on torch >= 1.4 do these unused methods get removed
        assert not callable(getattr(script, "training_step", None))


@pytest.mark.parametrize("modelclass", [
    EvalModelTemplate,
    ParityModuleRNN,
    BasicGAN,
])
@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.5.0"),
    reason="torch.save/load has bug loading script modules on torch <= 1.4",
)
def test_torchscript_save_load(tmpdir, modelclass):
    """ Test that scripted LightningModules is correctly saved and can be loaded. """
    model = modelclass()
    output_file = str(tmpdir / "model.pt")
    script = model.to_torchscript(file_path=output_file)
    loaded_script = torch.jit.load(output_file)
    assert torch.allclose(next(script.parameters()), next(loaded_script.parameters()))
