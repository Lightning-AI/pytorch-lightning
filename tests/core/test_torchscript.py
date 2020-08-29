import torch

from pytorch_lightning import LightningModule


class SimpleModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))


def test_torchscript_save_load(tmpdir):
    """ Test that scripted LightningModule behaves like the original. """
    model = SimpleModel()
    example_input = torch.rand(5, 64)
    script = model.to_torchscript()
    assert isinstance(script, torch.jit.ScriptModule)
    output_file = str(tmpdir / "model.jit")
    torch.jit.save(script, output_file)
    script = torch.jit.load(output_file)
    model_output = model(example_input)
    script_output = script(example_input)
    assert torch.allclose(script_output, model_output)
