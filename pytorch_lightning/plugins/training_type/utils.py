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
from pytorch_lightning.strategies.utils import on_colab_kaggle as _on_colab_kaggle
from pytorch_lightning.utilities import rank_zero_deprecation


def on_colab_kaggle() -> bool:
    rank_zero_deprecation(
        "`pl.plugins.training_type.utils.on_colab_kaggle` is deprecated in v1.6 and will be removed in v1.8."
        " Use `pl.strategies.utils.on_colab_kaggle` instead."
    )
    return _on_colab_kaggle()
