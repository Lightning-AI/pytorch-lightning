import traceback
from functools import wraps


def data_loader(fn):
    """Decorator to make any fx with this use the lazy property.

    :param fn:
    :return:
    """
    def inner_fx(self):
        return fn(self)
    return inner_fx
