# Copyright The PyTorch Lightning team.
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
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn

rank_zero_deprecation(
    "Using `pytorch_lightning.core.decorators.parameter_validation` is deprecated in v1.5, "
    "and will be removed in v1.7. It has been replaced by automatic parameters tying with "
    "`pytorch_lightning.utilities.params_tying.set_shared_parameters`"
)

from functools import wraps  # noqa: E402
from typing import Callable  # noqa: E402


def parameter_validation(fn: Callable) -> Callable:
    """Validates that the module parameter lengths match after moving to the device. It is useful when tying
    weights on TPU's.

    Args:
        fn: ``model_to_device`` method

    Note:
        TPU's require weights to be tied/shared after moving the module to the device.
        Failure to do this results in the initialization of new weights which are not tied.
        To overcome this issue, weights should be tied using the ``on_post_move_to_device`` model hook
        which is called after the module has been moved to the device.

    See Also:
        - `XLA Documentation <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks>`_
    """

    @wraps(fn)
    def inner_fn(self, *args, **kwargs):
        pre_layer_count = len(list(self.model.parameters()))
        module = fn(self, *args, **kwargs)
        self.model.on_post_move_to_device()
        post_layer_count = len(list(self.model.parameters()))

        if not pre_layer_count == post_layer_count:
            rank_zero_warn(
                "The model layers do not match after moving to the target device."
                " If your model employs weight sharing on TPU,"
                " please tie your weights using the `on_post_move_to_device` model hook.\n"
                f"Layer count: [Before: {pre_layer_count} After: {post_layer_count}]"
            )

        return module

    return inner_fn
