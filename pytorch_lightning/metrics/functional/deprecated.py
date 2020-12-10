from pytorch_lightning.metrics.utils import to_categorical as new_to_categorical
from pytorch_lightning.metrics.utils import to_onehot as new_to_onehot

import warnings


# TODO: remove in 1.1.1
def to_onehot(*args, **kwargs):
    warnings.warn(
        (
            "pytorch_lightning.metrics.functional.to_onehot is deprecated, "
            "please use pytorch_lightning.metrics.utils.to_onehot"
        ),
        DeprecationWarning,
    )
    return new_to_onehot(*args, **kwargs)


# TODO: remove in 1.1.1
def to_categorical(*args, **kwargs):
    warnings.warn(
        (
            "pytorch_lightning.metrics.functional.to_categorical is deprecated, "
            "please use pytorch_lightning.metrics.utils.to_categorical"
        ),
        DeprecationWarning,
    )
    return new_to_categorical(*args, **kwargs)
