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
from lightning.fabric.plugins.precision.bitsandbytes import BitsandbytesPrecision as FabricBNBPrecision
from lightning.pytorch.plugins.precision.precision import Precision


class BitsandbytesPrecision(Precision, FabricBNBPrecision):
    """Plugin for quantizing weights with `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    .. note::
        The optimizer is not automatically replaced with ``bitsandbytes.optim.Adam8bit`` or equivalent 8-bit optimizers.

    Args:
        mode: The quantization mode to use.
        dtype: The compute dtype to use.
        ignore_modules: The submodules whose Linear layers should not be replaced, for example. ``{"lm_head"}``.
            This might be desirable for numerical stability. The string will be checked in as a prefix, so a value like
            "transformer.blocks" will ignore all linear layers in all of the transformer blocks.
    """
