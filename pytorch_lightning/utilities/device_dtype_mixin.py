from pytorch_lightning.utilities import rank_zero_deprecation

rank_zero_deprecation(
    "`pytorch_lightning.utilities.device_dtype_mixin` has been moved to"
    " `pytorch_lightning.core.mixins.device_dtype_mixin` since v1.4"
    " and will be removed in v1.6."
)

# To support backward compatibility as `device_dtype_mixin` has been
# moved to `pytorch_lightning.core.mixins.device_dtype_mixin`
from pytorch_lightning.core.mixins.device_dtype_mixin import DeviceDtypeModuleMixin  # noqa: E402, F401 # isort: skip
