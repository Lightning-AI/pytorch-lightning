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
from pytorch_lightning.utilities.enums import _AcceleratorType, GradClipAlgorithmType, PrecisionType


def test_consistency():
    assert _AcceleratorType.TPU not in ("GPU", "CPU")
    assert _AcceleratorType.TPU in ("TPU", "CPU")
    assert _AcceleratorType.TPU in ("tpu", "CPU")
    assert _AcceleratorType.TPU not in {"GPU", "CPU"}
    # hash cannot be case invariant
    assert _AcceleratorType.TPU not in {"TPU", "CPU"}
    assert _AcceleratorType.TPU in {"tpu", "CPU"}


def test_precision_supported_types():
    assert PrecisionType.supported_types() == ["16", "32", "64", "bf16", "mixed"]
    assert PrecisionType.supported_type(16)
    assert PrecisionType.supported_type("16")
    assert not PrecisionType.supported_type(1)
    assert not PrecisionType.supported_type("invalid")


def test_gradient_clip_algorithms():
    assert GradClipAlgorithmType.supported_types() == ["value", "norm"]
    assert GradClipAlgorithmType.supported_type("norm")
    assert GradClipAlgorithmType.supported_type("value")
    assert not GradClipAlgorithmType.supported_type("norm2")
