"""General utilities"""
from enum import Enum

import numpy

from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.imports import *
from pytorch_lightning.utilities.parsing import AttributeDict, flatten_dict


# backward compatible
# TODO: remove in v1.0
APEX_AVAILABLE = is_apex_available()
NATIVE_AMP_AVALAIBLE = is_native_amp_available()

FLOAT16_EPSILON = numpy.finfo(numpy.float16).eps
FLOAT32_EPSILON = numpy.finfo(numpy.float32).eps
FLOAT64_EPSILON = numpy.finfo(numpy.float64).eps


class AMPType(Enum):
    APEX = 'apex'
    NATIVE = 'native'
