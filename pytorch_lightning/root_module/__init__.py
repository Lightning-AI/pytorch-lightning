"""
.. warning:: `root_module` package has been renamed to `core` since v0.6.0 and will be removed in v0.8.0
"""

import warnings

warnings.warn("`root_module` package has been renamed to `core` since v0.6.0"
              " and will be removed in v0.8.0", DeprecationWarning)

from pytorch_lightning.core import (  # noqa: E402
    decorators, grads, hooks, root_module, memory, model_saving)
