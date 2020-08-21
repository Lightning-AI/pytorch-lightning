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

import torch


def cast_model_to_precision(model):
    if model.precision in [16, 32]:
        model.to(dtype=torch.float32)
    elif model.precision == 64:
        model.to(dtype=torch.float64)
    else:
        raise Exception("unexpected precision")
    return model
