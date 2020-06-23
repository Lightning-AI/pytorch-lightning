"""
Models and datasets for PL testing
==================================

This is a supplementary package serving for setting common test cases and simplified versions of standard datasets.

"""

from pl_develop.model_template import EvalModelTemplate
from pl_develop.datasets import TrialMNIST, TrialCIFAR10
from pl_develop.models import EvalModelGAN, ParityModuleMNIST, ParityModuleRNN

__all__ = [
    'EvalModelTemplate',
    'TrialMNIST',
    'TrialCIFAR10',
    'EvalModelGAN',
    'ParityModuleMNIST',
    'ParityModuleRNN',
]
