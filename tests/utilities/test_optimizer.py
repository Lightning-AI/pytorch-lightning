import collections

import torch

from pytorch_lightning.utilities.optimizer import optimizer_to_device


def test_optimizer_to_device():
    """Ensure that after the initial seed everything, the seed stays the same for the same run."""
    layer = torch.nn.Linear(32, 2)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    import logging

    logging.error(f"Type of opt: {opt}, {type(opt)}")
    optimizer_to_device(opt, "cpu")
    if torch.cuda.is_available():
        layer.to("cuda")
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
