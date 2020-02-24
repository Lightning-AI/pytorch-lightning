import warnings


def data_loader(fn):
    """Decorator to make any fx with this use the lazy property.

    :param fn:
    :return:
    """
    w = 'data_loader decorator was deprecated in 0.6.1 and will be removed in 0.8.0'
    warnings.warn(w)

    value = fn()
    return value
