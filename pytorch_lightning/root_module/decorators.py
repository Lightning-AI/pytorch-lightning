from builtins import property as _property # Prevents recursive call.

def property(fn):
    # Warning: This shadows the usual property, so @property use
    # below this definition is invalid unless the safe variant is needed.
    """
    Makes properties in subclasses of nn.Module (e.g. LightningModule)
     return more informative AttributeError exceptions.
    :param fn:
    :return:
    """
    @_property
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AttributeError as e:
            raise RuntimeError('An AttributeError was encountered: ' + str(e)) from e
    return wrapper

def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    attr_name = '_lazy_' + fn.__name__

    @property
    def _data_loader(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _data_loader
