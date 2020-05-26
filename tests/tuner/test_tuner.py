import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate

def test_call_order(tmpdir):
    """ Check that an warning occurs if the methods are called in a different
        order than expected """
        
    