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
"""Helper functions to operate on metric values."""

from typing import Any

from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars


def metrics_to_scalars(metrics: Any) -> Any:
    """Recursively walk through a collection and convert single-item tensors to scalar values.

    Raises:
        ValueError:
            If tensors inside ``metrics`` contains multiple elements, hence preventing conversion to a scalar.
    """

    return convert_tensors_to_scalars(metrics)
