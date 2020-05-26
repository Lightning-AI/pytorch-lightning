from argparse import Namespace


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    >>> strtobool('YES')
    1
    >>> strtobool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError(f'invalid truth value {val}')


def clean_namespace(hparams):
    """
    Removes all functions from hparams so we can pickle
    :param hparams:
    :return:
    """

    if isinstance(hparams, Namespace):
        del_attrs = []
        for k in hparams.__dict__:
            if callable(getattr(hparams, k)):
                del_attrs.append(k)

        for k in del_attrs:
            delattr(hparams, k)

    elif isinstance(hparams, dict):
        del_attrs = []
        for k, v in hparams.items():
            if callable(v):
                del_attrs.append(k)

        for k in del_attrs:
            del hparams[k]


def nested_hasattr(namespace, attribute):
    """
    Recursively check if a namespace has a certain attribute
    """
    parts = attribute.split(".")
    for part in parts:
        if hasattr(namespace, part):
            namespace = getattr(namespace, part)
        else:
            return False
    else:
        return True


def nested_setattr(namespace, attribute, value):
    """
    Recursively look through a namespace for a certain attribute and set
    to a given value
    """
    parts = attribute.split(".")
    for part in parts[:-1]:
        if hasattr(namespace, part):
            namespace = getattr(namespace, part)
    setattr(namespace, parts[-1], value)
    
    
def nested_getattr(namespace, attribute):
    """
    Recursively look through a namespace for certain attribute and return
    the given value
    """
    parts = attribute.split(".")
    found = False
    for part in parts:
        if hasattr(namespace, part):
            namespace = getattr(namespace, part)
        else:
            found = False
    else:
        found = True
    
    if found:    
        return namespace
    else:
        raise AttributeError(f'{namespace} object has no attribute {attribute}')
