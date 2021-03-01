from warnings import warn

warn(
    "`swa` package has been renamed to `stochastic_weight_avg` since v1.3 and will be removed in v1.5",
    DeprecationWarning
)

from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging  # noqa: F401 E402
