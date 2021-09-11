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
from typing import List, Union

from pytorch_lightning.callbacks import ModelSummary


class RichModelSummary(ModelSummary):
    @staticmethod
    def summarize(
        summary_data: List[List[Union[str, List[str]]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
    ) -> None:
        pass
