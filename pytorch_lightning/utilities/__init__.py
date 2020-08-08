"""General utilities"""

import numpy

from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.parsing import AttributeDict, flatten_dict


FLOAT16_EPSILON = numpy.finfo(numpy.float16).eps
FLOAT32_EPSILON = numpy.finfo(numpy.float32).eps
FLOAT64_EPSILON = numpy.finfo(numpy.float64).eps
