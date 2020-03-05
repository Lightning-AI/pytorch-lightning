
from abc import ABC

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
import logging as log


class TrainerAMPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    use_amp: bool

    def init_amp(self, use_amp):
        self.use_amp = use_amp and APEX_AVAILABLE
        if self.use_amp:
            log.info('Using 16bit precision.')

        if use_amp and not APEX_AVAILABLE:  # pragma: no cover
            msg = """
            You set `use_amp=True` but do not have apex installed.
            Install apex first using this guide and rerun with use_amp=True:
            https://github.com/NVIDIA/apex#linux

            this run will NOT use 16 bit precision
            """
            raise ModuleNotFoundError(msg)
