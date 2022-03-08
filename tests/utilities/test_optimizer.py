import collections

import torch

from pytorch_lightning.utilities.optimizer import optimizer_to_device


def test_optimizer_to_device():
    class TestOptimizer(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.state["dummy"] = torch.tensor(0)

    layer = torch.nn.Linear(32, 2)
    opt = TestOptimizer(layer.parameters(), lr=0.1)
    optimizer_to_device(opt, "cpu")
    if torch.cuda.is_available():
        optimizer_to_device(opt, "cuda")
        assert_opt_parameters_on_device(opt, "cuda")


def assert_opt_parameters_on_device(opt, device: str):
    for param in opt.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            assert param.data.device.type == device
        elif isinstance(param, collections.Mapping):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    assert param.data.device.type == device
