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
import pytest

from pytorch_lightning.utilities.enums import DeviceType, ModelSummaryMode, PrecisionType


def test_consistency():
    assert DeviceType.TPU not in ("GPU", "CPU")
    assert DeviceType.TPU in ("TPU", "CPU")
    assert DeviceType.TPU in ("tpu", "CPU")
    assert DeviceType.TPU not in {"GPU", "CPU"}
    # hash cannot be case invariant
    assert DeviceType.TPU not in {"TPU", "CPU"}
    assert DeviceType.TPU in {"tpu", "CPU"}


def test_precision_supported_types():
    assert PrecisionType.supported_types() == ["16", "32", "64", "bf16"]
    assert PrecisionType.supported_type(16)
    assert PrecisionType.supported_type("16")
    assert not PrecisionType.supported_type(1)
    assert not PrecisionType.supported_type("invalid")


def test_model_summary_mode():
    assert ModelSummaryMode.supported_types() == ["top", "full"]
    assert ModelSummaryMode.TOP in ("top", "full")
    assert ModelSummaryMode.get_max_depth("top") == 1
    assert ModelSummaryMode.get_max_depth("full") == -1

    with pytest.raises(ValueError, match=f"`mode` can be {', '.join(list(ModelSummaryMode))}, got invalid."):
        ModelSummaryMode.get_max_depth("invalid")
