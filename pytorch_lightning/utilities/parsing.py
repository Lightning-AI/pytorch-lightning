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


def lightning_hasattr(model, attribute):
    """ Special hasattr for lightning. Checks for attribute in model namespace
        and the old hparams namespace/dict """
    # New format
    if hasattr(model, attribute):
        return True
    # Old hparams format, either namespace or dict
    elif hasattr(model, 'hparams'):
        if isinstance(model.hparams, dict):
            if attribute in model.hparams:
                return True
        else:
            if hasattr(model.hparams, attribute):
                return True
    else:
        return False        


def lightning_getattr(model, attribute):
    """ Special getattr for lightning. Checks for attribute in model namespace
        and the old hparams namespace/dict """
    # New format
    if hasattr(model, attribute):
        return getattr(model, attribute)
    # Old hparams format, either namespace or dict
    elif hasattr(model, 'hparams'):
        if isinstance(model.hparams, dict):
            return model.hparams[attribute]
        else:
            return getattr(model.hparams, attribute)
    else:
        raise ValueError(f'{attribute} is not stored in the model namespace'
                          ' or the `hparams` namespace/dict.')


def lightning_setattr(model, attribute, value):
    """ Special setattr for lightning. Checks for attribute in model namespace
        and the old hparams namespace/dict """
    # New format
    if hasattr(model, attribute):
        setattr(model, attribute, value)
        return
    # Old hparams format, either namespace or dict
    elif hasattr(model, 'hparams'):
        if isinstance(model.hparams, dict):
            model.hparams[attribute] = value
            return
        else:
            setattr(model.hparams, attribute, value)
    else:
        raise ValueError(f'{attribute} is not stored in the model namespace'
                          ' or the `hparams` namespace/dict.')