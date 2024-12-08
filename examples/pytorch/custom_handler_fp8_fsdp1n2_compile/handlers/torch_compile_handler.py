import logging
import operator

import torch
import torch.nn as nn
from lightning_utilities.core.imports import compare_version

log = logging.getLogger(__name__)


class TorchCompileHandler:
    """Handler for compiling specific layers of the model using torch.compile.

    Args:
        enable_compile (bool): Whether to enable compilation.
        model_path (str): Path to the model, used to determine default compilable layers.
        compile_layers (List[str], optional): List of layer class names to compile. If None, defaults are used.
        compile_args (dict, optional): Additional arguments to pass to torch.compile.

    """

    # Default mapping of model names to compilable layer class names
    DEFAULT_COMPILABLE_LAYERS = {
        "llama": ["LlamaMLP"],
        "mixtral": ["MixtralBlockSparseTop2MLP"],
    }

    def __init__(
        self,
        enable_compile: bool,
        model_path: str,
        compile_layers: list = None,
        compile_args: dict = None,
    ):
        self.enable_compile = enable_compile
        self.model_path = model_path.lower()
        self.compile_args = compile_args if compile_args is not None else {}
        self.compile_layers = compile_layers  # User-provided layers to compile

        if self.enable_compile:
            # Check PyTorch version for torch.compile support (requires PyTorch >= 2.6.0)
            try:
                compare_version("torch", operator.ge, "2.6.0")
            except RuntimeError as e:
                log.error(str(e))
                raise

            # Determine default layers to compile if not provided
            if self.compile_layers is None:
                self.compile_layers = self._get_default_compile_layers()
                if not self.compile_layers:
                    log.warning(
                        "No default compilable layers found for the model. " "Please provide compile_layers explicitly."
                    )

    def _get_default_compile_layers(self):
        """Determines the default layers to compile based on the model name.

        Returns:
            List[str]: List of layer class names to compile.

        """
        for model_name, layers in self.DEFAULT_COMPILABLE_LAYERS.items():
            if model_name in self.model_path:
                return layers
        return []

    def compile_model(self, model: nn.Module):
        """Compiles specified layers in the model.

        Args:
            model (nn.Module): The model to compile.

        """
        if not self.enable_compile:
            return

        if not self.compile_layers:
            log.warning("No layers specified for compilation. Skipping compilation.")
            return

        log.warning(f"Compiling layers: {self.compile_layers} with args: {self.compile_args}")

        self._compile_layers(model)

    def _compile_layers(self, module: nn.Module):
        """Recursively compiles specified layers in the module.

        Args:
            module (nn.Module): The module to process.

        """
        for name, child in module.named_children():
            child_class_name = type(child).__name__
            if child_class_name in self.compile_layers:
                log.warning(f"Compiling layer {name} ({child_class_name})")
                try:
                    # Compile the layer with provided arguments
                    compiled_child = torch.compile(child, **self.compile_args)
                    setattr(module, name, compiled_child)
                except Exception as e:
                    log.error(f"Failed to compile layer {name}: {e}")
                    raise
            else:
                # Recursively process child modules
                self._compile_layers(child)
