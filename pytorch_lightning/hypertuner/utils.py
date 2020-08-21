from typing import Callable
from functools import wraps

from pytorch_lightning.utilities import rank_zero_warn

def check_call_order(method: Callable):
    """ Define decorator that checks for the correct call order """
    @wraps(method)
    def check_order(self, *method_args, **method_kwargs):
        method_name = method.__name__
        methods = self.call_order[method_name]
        for m in methods:
            if getattr(self, '_' + m + '_called'):
                rank_zero_warn(f'The results of `{method_name}` will influence'
                               f' the results of {methods}. You have already called'
                               f' `{m}` which should have been called after {method_name}.')

        setattr(self, '_' + method.__name__ + '_called', True)
        return method(self, *method_args, **method_kwargs)

    return check_order