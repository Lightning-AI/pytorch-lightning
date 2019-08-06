"""
Module to describe gradients
"""

from torch import nn


class GradInformation(nn.Module):

    def grad_norm(self, norm_type):
        results = {}
        total_norm = 0
        for i, p in enumerate(self.parameters()):
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)

                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    results['grad_{}_norm_{}'.format(norm_type, i)] = grad
                except Exception:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1. / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        results['grad_{}_norm_total'.format(norm_type)] = grad
        return results
