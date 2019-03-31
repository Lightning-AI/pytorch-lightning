import numpy as np
from torch import nn

"""
Module to describe gradients
"""


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

                    results['grad_{}_norm_{}'.format(norm_type, i)] = round(norm.data.cpu().numpy().flatten()[0], 3)
                except Exception as e:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1. / norm_type)
        results['grad_{}_norm_total'.format(norm_type)] = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        return results


    def describe_grads(self):
        for p in self.parameters():
            g = p.grad.data.numpy().flatten()
            print(np.max(g), np.min(g), np.mean(g))


    def describe_params(self):
        for p in self.parameters():
            g = p.data.numpy().flatten()
            print(np.max(g), np.min(g), np.mean(g))