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
"""
Module to describe gradients
"""
from typing import Dict, Union

import torch
from torch.nn import Module


class GradInformation(Module):

    def grad_norm(self, norm_type: Union[float, int, str]) -> Dict[str, float]:
        """Compute each parameter's gradient's norm and their overall norm.

        The overall norm is computed over all gradients together, as if they
        were concatenated into a single vector.

        Args:
            norm_type: The type of the used p-norm, cast to float if necessary.
                Can be ``'inf'`` for infinity norm.

        Return:
            norms: The dictionary of p-norms of each parameter's gradient and
                a special entry for the total p-norm of the gradients viewed
                as a single vector.
        """
        norm_type = float(norm_type)

        norms, all_norms = {}, []
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            param_norm = float(p.grad.data.norm(norm_type))
            norms[f'grad_{norm_type}_norm_{name}'] = round(param_norm, 4)

            all_norms.append(param_norm)

        total_norm = float(torch.tensor(all_norms).norm(norm_type))
        norms[f'grad_{norm_type}_norm_total'] = round(total_norm, 4)

        return norms
