import traceback
from functools import wraps
import warnings


def data_loader(fn):
    """Decorator to make any fx with this use the lazy property.

    :param fn:
    :return:
    """
    w = 'data_loader decorator deprecated in 0.7.0. Will remove 0.9.0'
    warnings.warn(w)

    def inner_fx(self):
        return fn(self)
    return inner_fx
