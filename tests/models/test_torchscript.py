import pytest
import torch

from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule
from tests.base.models import ParityModuleRNN, TestGAN


@pytest.mark.parametrize("modelclass", [
    EvalModelTemplate,
    ParityModuleRNN,
    TestGAN,
])
def test_torchscript_input_output(modelclass):
    """ Test that scripted LightningModule forward works. """
    model = modelclass()
    script = model.to_torchscript()
    assert isinstance(script, torch.jit.ScriptModule)
    model_output = model(model.example_input_array)
    script_output = script(model.example_input_array)
    assert torch.allclose(script_output, model_output)


@pytest.mark.parametrize("modelclass", [
    EvalModelTemplate,
    ParityModuleRNN,
    TestGAN,
])
def test_torchscript_properties(modelclass):
    """ Test that scripted LightningModule has unnecessary methods removed. """
    model = modelclass()
    model.datamodule = TrialMNISTDataModule()
    script = model.to_torchscript()
    assert not hasattr(script, "datamodule")
    assert not hasattr(model, "batch_size") or hasattr(script, "batch_size")
    assert not hasattr(model, "learning_rate") or hasattr(script, "learning_rate")
    assert not callable(getattr(script, "training_step", None))


@pytest.mark.parametrize("modelclass", [
    EvalModelTemplate,
    ParityModuleRNN,
    TestGAN,
])
def test_torchscript_save_load(tmpdir, modelclass):
    """ Test that scripted LightningModules can be saved and loaded. """
    model = modelclass()
    script = model.to_torchscript()
    assert isinstance(script, torch.jit.ScriptModule)
    output_file = str(tmpdir / "model.jit")
    torch.jit.save(script, output_file)
    torch.jit.load(output_file)
