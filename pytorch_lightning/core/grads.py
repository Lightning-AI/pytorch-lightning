"""
Module to describe gradients
"""
from typing import Dict, Union

import torch


class GradInformation(torch.nn.Module):

    def grad_norm(self, norm_type: Union[float, int, str]) -> Dict[str, float]:
        r"""Compute individual parameter's gradient norms and the overall norm.

        The overall norm is computed over all gradients together, as if they
        were concatenated into a single vector.

        Args:
            norm_type: The type of the used p-norm, cast to float if necessary.
                Can be ``'inf'`` for infinity norm.

        Return:
            norms: The dictionary of p-norms each individual gradient and the a
                special entry for the total p-norm of the parameters' gradients
                viewed as a single vector.
        """
        norm_type = float(norm_type)

        norms, all_norms = {}, []
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            param_norm = float(p.grad.data.norm(norm_type))
            norms[f'grad_{norm_type}_norm_{name}'] = round(param_norm, 3)

            all_norms.append(param_norm)

        total_norm = float(torch.tensor(all_norms).norm(norm_type))
        norms[f'grad_{norm_type}_norm_total'] = round(total_norm, 3)

        return norms
