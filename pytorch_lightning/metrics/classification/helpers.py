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
from typing import Optional, Tuple

import numpy as np
import torch
from torchmetrics.classification.checks import _check_classification_inputs
from torchmetrics.utilities.data import select_topk, to_onehot

from pytorch_lightning.utilities import LightningEnum


class DataType(LightningEnum):
    """
    Enum to represent data type
    """

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


class AverageMethod(LightningEnum):
    """
    Enum to represent average method
    """

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = "none"
    SAMPLES = "samples"


class MDMCAverageMethod(LightningEnum):
    """
    Enum to represent multi-dim multi-class average method
    """

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"