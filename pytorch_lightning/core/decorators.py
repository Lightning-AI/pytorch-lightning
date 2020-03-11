import traceback
from functools import wraps
import warnings


def data_loader(fn):
    """Decorator to make any fx with this use the lazy property.

    Warnings:
        This decorator deprecated in v0.7.0 and it will be removed v0.9.0.
    """
    warnings.warn('`data_loader` decorator deprecated in v0.7.0. Will be removed v0.9.0', DeprecationWarning)

    def inner_fx(self):
        return fn(self)
    return inner_fx
