try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class TrainerAMPMixin(object):

    def init_amp(self, use_amp):
        self.use_amp = use_amp and APEX_AVAILABLE
        if self.use_amp:
            print('using 16bit precision')

        if use_amp and not APEX_AVAILABLE:  # pragma: no cover
            msg = """
            You set use_amp=True but do not have apex installed.
            Install apex first using this guide and rerun with use_amp=True:
            https://github.com/NVIDIA/apex#linux

            this run will NOT use 16 bit precision
            """
            raise ModuleNotFoundError(msg)
