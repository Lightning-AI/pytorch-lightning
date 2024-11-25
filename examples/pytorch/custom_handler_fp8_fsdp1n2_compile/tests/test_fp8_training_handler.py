import unittest
from unittest.mock import patch

import torch.nn as nn
from handlers.fp8_training_handler import Float8TrainingHandler, FP8Config
from lightning.pytorch.demos import Transformer
from torchao.float8 import Float8Linear


class TestFloat8TrainingHandler(unittest.TestCase):
    def setUp(self):
        self.args = FP8Config(
            enable_fp8=True,
            enable_amax_init=True,
            scaling_type_input="delayed",
            scaling_type_weight="delayed",
            scaling_type_grad_output="delayed",
            enable_fsdp_float8_all_gather=False,
            precompute_float8_dynamic_scale_for_fsdp=False,
            pad_inner_dim=False,
            emulate_fp8=False,  # Set to True for testing without FP8 hardware
            enable_torch_compile=False,
            enable_pre_and_post_forward=False,
        )

        self.model_path = "test_mixtral_model"
        self.parallel_dims = {"dp_shard_enabled": False}

        # Simple model for testing
        self.model = Transformer(
            vocab_size=32000,
            nlayers=16,
            nhid=4096,
            ninp=1024,
            nhead=32,
        )

    @patch("handlers.fp8_training_handler.is_sm89_or_later", return_value=True)
    def test_handler_initialization(self, mock_sm89):
        handler = Float8TrainingHandler(self.args, self.model_path, self.parallel_dims)
        self.assertTrue(handler.enable_fp8)
        self.assertFalse(handler.compile)
        self.assertIsNotNone(handler.args)
        self.assertIsNotNone(handler.parallel_dims)

    @patch("handlers.fp8_training_handler.is_sm89_or_later", return_value=True)
    def test_compile_flag(self, mock_sm89):
        self.args.enable_torch_compile = True
        handler = Float8TrainingHandler(self.args, self.model_path, self.parallel_dims)
        self.assertTrue(handler.compile)

    @patch("handlers.fp8_training_handler.is_sm89_or_later", return_value=False)
    def test_handler_disabled_on_unsupported_hardware(self, mock_sm89):
        # Assert that the RuntimeError is raised
        with self.assertRaises(RuntimeError) as context:
            Float8TrainingHandler(self.args, self.model_path, self.parallel_dims)

        # Check that the error message matches the expected text
        self.assertIn(
            "Float8Linear operation is not supported on the current hardware.",
            str(context.exception),
        )

    def test_handler_disabled_when_fp8_not_enabled(self):
        self.args.enable_fp8 = False
        handler = Float8TrainingHandler(self.args, self.model_path, self.parallel_dims)
        self.assertFalse(handler.enable_fp8)

    @patch("handlers.fp8_training_handler.is_sm89_or_later", return_value=True)
    def test_convert_to_float8_training(self, mock_sm89):
        handler = Float8TrainingHandler(self.args, self.model_path, self.parallel_dims)
        handler.convert_to_float8_training(self.model)

        # Check if nn.Linear layers have been converted to Float8Linear
        print(self.model)
        for module_name, module in self.model.named_modules():
            if any(proj in module_name for proj in ["w1", "w2", "w3"]):  # Float8Linear
                self.assertIsInstance(module, Float8Linear, f"{module_name} should be Float8Linear")
            elif isinstance(module, nn.Linear):
                self.assertNotIsInstance(module, Float8Linear, f"{module_name} should not be Float8Linear")

    @patch("handlers.fp8_training_handler.is_sm89_or_later", return_value=True)
    def test_precompute_float8_dynamic_scale_for_fsdp(self, mock_sm89):
        handler = Float8TrainingHandler(self.args, self.model_path, self.parallel_dims)
        handler.convert_to_float8_training(self.model)

        with patch("torchao.float8.precompute_float8_dynamic_scale_for_fsdp") as mock_precompute:
            handler.precompute_float8_dynamic_scale_for_fsdp(self.model)
            mock_precompute.assert_not_called()  # Should not be called since precompute_scale is False

        # Enable precompute_scale
        args = self.args
        args.scaling_type_input = "dynamic"
        args.scaling_type_weight = "dynamic"
        args.scaling_type_grad_output = "dynamic"
        args.enable_fsdp_float8_all_gather = True
        args.precompute_float8_dynamic_scale_for_fsdp = True
        handler = Float8TrainingHandler(args, self.model_path, {"dp_shard_enabled": True})
        model = Transformer(
            vocab_size=32000,
            nlayers=16,
            nhid=4096,
            ninp=1024,
            nhead=32,
        )  # recreate the model
        with patch("torchao.float8.precompute_float8_dynamic_scale_for_fsdp") as mock_precompute:
            handler.precompute_float8_dynamic_scale_for_fsdp(model)
            mock_precompute.assert_called()

    @patch("handlers.fp8_training_handler.is_sm89_or_later", return_value=True)
    def test_sync_float8_amax_and_scale_history(self, mock_sm89):
        handler = Float8TrainingHandler(self.args, self.model_path, self.parallel_dims)
        handler.convert_to_float8_training(self.model)

        with patch("torchao.float8.sync_float8_amax_and_scale_history") as mock_sync:
            handler.sync_float8_amax_and_scale_history(self.model)
            mock_sync.assert_called_once_with(self.model)
