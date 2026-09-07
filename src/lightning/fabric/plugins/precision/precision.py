# Copyright The Lightning AI team.
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
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from lightning.fabric.utilities.types import _PARAMETERS, Optimizable

_PRECISION_INPUT_INT = Literal[64, 32, 16]
_PRECISION_INPUT_STR_ALIAS_CONVERSION = {"64": "64-true", "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"}
_PRECISION_INPUT_STR_ALIAS = Literal["64", "32", "16", "bf16"]
_PRECISION_INPUT_STR = Literal[
    "transformer-engine",
    "transformer-engine-float16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
]
_PRECISION_INPUT = Union[_PRECISION_INPUT_INT, _PRECISION_INPUT_STR, _PRECISION_INPUT_STR_ALIAS]


class Precision:
    """Base class for all plugins handling the precision-specific parts of the training.

    The class attribute precision must be overwritten in child classes. The default value reflects fp32 training.

    """

    precision: _PRECISION_INPUT_STR = "32-true"

    def convert_module(self, module: Module) -> Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        return module

    def tensor_init_context(self) -> AbstractContextManager:
        """Controls how tensors get created (device, dtype)."""
        return nullcontext()

    def module_init_context(self) -> AbstractContextManager:
        """Instantiate module parameters or tensors in the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        return nullcontext()

    def forward_context(self) -> AbstractContextManager:
        """A contextmanager for managing model forward/training_step/evaluation_step/predict_step."""
        return nullcontext()

    def convert_input(self, data: Any) -> Any:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        This is a no-op in the base precision plugin, since we assume the data already has the desired type (default is
        torch.float32).

        """
        return data

    def convert_output(self, data: Any) -> Any:
        """Convert outputs to the floating point precision type expected after model's forward.

        This is a no-op in the base precision plugin, since we assume the data already has the desired type (default is
        torch.float32).

        """
        return data

    def pre_backward(self, tensor: Tensor, module: Optional[Module]) -> Any:
        """Runs before precision plugin executes backward.

        Args:
            tensor: The tensor that will be used for backpropagation
            module: The module that was involved in producing the tensor and whose parameters need the gradients

        """

    def backward(self, tensor: Tensor, model: Optional[Module], *args: Any, **kwargs: Any) -> None:
        """Performs the actual backpropagation.

        Args:
            tensor: The tensor that will be used for backpropagation
            model: The module that was involved in producing the tensor and whose parameters need the gradients

        """
        tensor.backward(*args, **kwargs)

    def post_backward(self, tensor: Tensor, module: Optional[Module]) -> Any:
        """Runs after precision plugin executes backward.

        Args:
            tensor: The tensor that will be used for backpropagation
            module: The module that was involved in producing the tensor and whose parameters need the gradients

        """

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        """Hook to run the optimizer step."""
        return optimizer.step(**kwargs)

    def main_params(self, optimizer: Optimizer) -> _PARAMETERS:
        """The main params of the model.

        Returns the plain model params here. Maybe different in other precision plugins.

        """
        for group in optimizer.param_groups:
            yield from group["params"]

    def unscale_gradients(self, optimizer: Optimizer) -> None:
        return

    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint, implement to generate precision plugin state_dict.

        Returns:
            A dictionary containing precision plugin state.

        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload precision plugin state given precision plugin
        state_dict.

        Args:
            state_dict: the precision plugin state returned by ``state_dict``.

        """
        pass

    def teardown(self) -> None:
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.

        """
