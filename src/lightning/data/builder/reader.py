# Copyright The Lightning AI team.
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

from typing import Any, Dict, Optional
from lightning_utilities.core.imports import RequirementCache
from lightning.data.builder.serializers import _SERIALIZERS
from lightning.data.builder.base import BaseWriter
import numpy as np
import json
import os

class Reader:
    
    def __init__(self, out_dir: str):
        super().__init__()

        self.out_dir = out_dir

    def read(self, index: int, rank):
        pass

