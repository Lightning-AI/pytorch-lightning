from abc import ABC

from pytorch_lightning import _logger as log

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

    def init_amp(self, use_amp):
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
        return self.precision == 16 and APEX_AVAILABLE
