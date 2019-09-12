from functools import wraps


def rank_zero_only(fn):
    """Decorate a logger method to run it only on the process with rank 0"""
    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.rank == 0:
            fn(self, *args, **kwargs)
    return wrapped_fn


class LightningLoggerBase:
    
    def __init__(self):
        self._rank = 0

    def log_metrics(self, metrics, step_num):
        raise NotImplementedError()
    
    def log_hyperparams(self, params):
        raise NotImplementedError()
    
    def save(self):
        pass

    def finalize(self, status):
        pass

    @property
    def rank(self):
        return self._rank
    
    @rank.setter
    def rank(self, value):
        self._rank = value
    
    @property
    def version(self):
        return None
