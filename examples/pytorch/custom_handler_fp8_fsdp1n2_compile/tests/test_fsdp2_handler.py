import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
from handlers.fsdp2_handler import FSDP2Config, FSDP2Handler


# Define mock functions
def mock_fully_shard(module, **kwargs):
    """Mock for torch.distributed._composable.fsdp.fully_shard.

    Returns the module unchanged to simulate sharding without actual processing.

    """
    return module


def mock_checkpoint_wrapper(module):
    """Mock for torch.distributed.algorithms._checkpoint.checkpoint_wrapper.

    Returns the module unchanged to simulate checkpoint wrapping without actual processing.

    """
    return module


class TestFSDP2Handler(unittest.TestCase):
    def setUp(self):
        self.args = FSDP2Config(
            enable_gradient_checkpointing=True,
            enable_cpu_offload=False,
        )

        # Mock device mesh
        self.device_mesh = {"data_parallel": MagicMock()}
        self.device_mesh["data_parallel"].size.return_value = 2  # Simulate more than one device

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model  # The wrapped Transformer model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        class InnerModel(nn.Module):
            def __init__(self, num_layers, input_size, hidden_size):
                super().__init__()
                # Initialize a ModuleList to store the layers
                self.layers = nn.ModuleList()
                for _ in range(num_layers):
                    layer = nn.Linear(input_size, hidden_size)
                    self.layers.append(layer)
                    # You can add more complex layers or custom layers here

            def forward(self, x):
                # Pass the input through each layer sequentially
                for layer in self.layers:
                    x = layer(x)
                return x

        self.model = ModelWrapper(
            InnerModel(
                num_layers=16,
                input_size=4096,
                hidden_size=1024,
            )
        )

    @patch("torch.distributed._composable.fsdp.fully_shard", side_effect=mock_fully_shard)
    @patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper",
        side_effect=mock_checkpoint_wrapper,
    )
    def test_wrap_model(self, mock_checkpoint_wrapper_func, mock_fully_shard_func):
        handler = FSDP2Handler(self.args, self.device_mesh)
        wrapped_model = handler.wrap_model(self.model)

        # Ensure fully_shard and checkpoint_wrapper are called
        assert mock_fully_shard_func.called, "fully_shard was not called"
        assert mock_checkpoint_wrapper_func.called, "checkpoint_wrapper was not called"

        # Verify that the model's layers have been wrapped
        assert wrapped_model is not None, "wrapped_model is None"
        mock_fully_shard_func.assert_called()

        # Ensure that checkpoint_wrapper is called for each layer
        assert mock_checkpoint_wrapper_func.call_count == len(self.model.model.layers)
        # Ensure that fully_shard is called for each layer + full module
        assert mock_fully_shard_func.call_count == len(self.model.model.layers) + 1

    def test_wrap_model_with_single_device(self):
        # Simulate single device
        self.device_mesh["data_parallel"].size.return_value = 1
        handler = FSDP2Handler(self.args, self.device_mesh)
        with pytest.raises(AssertionError):
            handler.wrap_model(self.model)

    @patch("torch.distributed._composable.fsdp.fully_shard", side_effect=mock_fully_shard)
    def test_enable_cpu_offload(self, mock_fully_shard_func):
        self.args.enable_cpu_offload = True
        handler = FSDP2Handler(self.args, self.device_mesh)
        handler.wrap_model(self.model)
        # Check if CPUOffloadPolicy is used
        args, kwargs = mock_fully_shard_func.call_args
        assert "offload_policy" in kwargs
        assert kwargs["offload_policy"] is not None

    @patch("torch.distributed._composable.fsdp.fully_shard", side_effect=mock_fully_shard)
    @patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper",
        side_effect=mock_checkpoint_wrapper,
    )
    def test_diable_gradient_checkpointing(self, mock_checkpoint_wrapper_func, mock_fully_shard_func):
        self.args.enable_gradient_checkpointing = False
        handler = FSDP2Handler(self.args, self.device_mesh)
        handler.wrap_model(self.model)
        # Check if gradient checkpointing is disabled
        assert not mock_checkpoint_wrapper_func.called, "Error: checkpoint_wrapper was unexpectedly called."
