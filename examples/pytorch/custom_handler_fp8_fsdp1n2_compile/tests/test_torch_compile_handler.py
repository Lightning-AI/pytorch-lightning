# test_torch_compile_handler.py

import unittest
from unittest.mock import patch

import torch.nn as nn
from handlers.torch_compile_handler import TorchCompileHandler
from lightning.pytorch.demos import Transformer


def mock_torch_compile(module, **kwargs):
    """Mock function for torch.compile that returns the module unchanged.

    This avoids actual compilation during testing.

    """
    return module


class TestTorchCompileHandler(unittest.TestCase):
    def setUp(self):
        self.enable_compile = True
        self.model_path = "test_custom_transformer_model"

        self.num_layers = 16
        self.model = Transformer(
            vocab_size=32000,
            nlayers=self.num_layers,
            nhid=4096,
            ninp=1024,
            nhead=32,
        )
        self.compile_args = {"backend": "inductor", "mode": "default"}

    @patch("torch.compile", side_effect=mock_torch_compile)
    def test_compile_transformer_encoder_layers(self, mock_compile):
        handler = TorchCompileHandler(
            enable_compile=self.enable_compile,
            model_path=self.model_path,
            compile_layers=["TransformerEncoderLayer"],  # Explicitly specify layers
            compile_args=self.compile_args,
        )
        handler.compile_model(self.model)

        # Ensure torch.compile was called with the correct layer
        assert mock_compile.call_count == self.num_layers, f"Expected mock_compile to be called {self.num_layers} times"

    def test_compile_disabled(self):
        handler = TorchCompileHandler(False, self.model_path)
        with patch("torch.compile") as mock_torch_compile:
            handler.compile_model(self.model)
            mock_torch_compile.assert_not_called()

    @patch("torch.compile", side_effect=mock_torch_compile)
    def test_compile_recursive(self, mock_compile):
        # Nested modules
        class NestedModel(nn.Module):
            def __init__(self, child_module):
                super().__init__()
                self.layer = nn.Sequential(
                    nn.Linear(128, 128),
                    child_module,
                )

            def forward(self, x):
                return self.layer(x)

        model = NestedModel(child_module=self.model)
        handler = TorchCompileHandler(self.enable_compile, self.model_path, compile_layers=["TransformerDecoderLayer"])
        handler.compile_model(model)

        # LlamaMLP inside NestedModel should be compiled
        assert mock_compile.called
        assert mock_compile.call_count == self.num_layers, f"Expected mock_compile to be called {self.num_layers} times"
