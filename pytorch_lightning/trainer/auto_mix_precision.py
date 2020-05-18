from abc import ABC
import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import rank_zero_warn

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True


class TrainerAMPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    precision: int
    use_native_amp: bool

    def init_amp(self, use_amp):
        # TODO: remove in v 0.8.0
        if self.use_native_amp:
            rank_zero_warn("`amp_level` has been deprecated since v0.7.4 "
                           "(native amp does not require it)"
                           " and this argument will be removed in v0.8.0", DeprecationWarning)

        # Backward compatibility, TODO: remove in v0.9.0
        if use_amp is not None:
            rank_zero_warn("`use_amp` has been replaced by `precision` since v0.7.0"
                           " and this argument will be removed in v0.9.0", DeprecationWarning)
            self.precision = 16 if use_amp else 32

        assert self.precision in (16, 32), 'only 32 or 16 bit precision supported'

        if use_amp and self.use_native_amp:
            log.info('Using 16bit precision.')
            return

        # TODO: remove all below for v0.8.0
        if use_amp and not APEX_AVAILABLE:  # pragma: no-cover
            raise ModuleNotFoundError("""
            You set `use_amp=True` but do not have apex installed.
            Install apex first using this guide and rerun with use_amp=True:
            https://github.com/NVIDIA/apex#linux

            this run will NOT use 16 bit precision
            """)

        if self.use_amp:
            log.info('Using 16bit precision.')

    @property
    def use_amp(self) -> bool:
        return self.precision == 16
