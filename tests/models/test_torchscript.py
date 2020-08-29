import torch

from pytorch_lightning import LightningModule
from tests.base import EvalModelTemplate


class SimpleModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))


def test_torchscript_save_load(tmpdir):
    """ Test that scripted LightningModule behaves like the original. """
    model = EvalModelTemplate()
    script = model.to_torchscript()
    assert isinstance(script, torch.jit.ScriptModule)
    output_file = str(tmpdir / "model.jit")
    torch.jit.save(script, output_file)
    script = torch.jit.load(output_file)
    # properties
    assert script.batch_size == model.batch_size
    assert script.learning_rate == model.learning_rate
    assert not callable(getattr(script, "training_step", None))
    # output matches
    model_output = model(model.example_input_array)
    script_output = script(model.example_input_array)
    assert torch.allclose(script_output, model_output)
