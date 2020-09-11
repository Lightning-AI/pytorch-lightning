import inspect
from argparse import ArgumentParser, Namespace
from typing import Union, List, Tuple, Any
from pytorch_lightning.utilities import parsing


def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
    """
    Create an instance from CLI arguments.

    Args:
        args: The parser or namespace to take arguments from. Only known arguments will be
            parsed and passed to the :class:`Trainer`.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
            These must be valid Trainer arguments.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> parser = ArgumentParser(add_help=False)
        >>> parser = Trainer.add_argparse_args(parser)
        >>> parser.add_argument('--my_custom_arg', default='something')  # doctest: +SKIP
        >>> args = Trainer.parse_argparser(parser.parse_args(""))
        >>> trainer = Trainer.from_argparse_args(args, logger=False)
    """
    if isinstance(args, ArgumentParser):
        args = cls.parse_argparser(args)
    params = vars(args)

    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(cls.__init__).parameters
    trainer_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
    trainer_kwargs.update(**kwargs)

    return cls(**trainer_kwargs)


def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
    """Parse CLI arguments, required for custom bool types."""
    args = arg_parser.parse_args() if isinstance(arg_parser, ArgumentParser) else arg_parser

    types_default = {
        arg: (arg_types, arg_default) for arg, arg_types, arg_default in get_init_arguments_and_types(cls)
    }

    modified_args = {}
    for k, v in vars(args).items():
        if k in types_default and v is None:
            # We need to figure out if the None is due to using nargs="?" or if it comes from the default value
            arg_types, arg_default = types_default[k]
            if bool in arg_types and isinstance(arg_default, bool):
                # Value has been passed as a flag => It is currently None, so we need to set it to True
                # We always set to True, regardless of the default value.
                # Users must pass False directly, but when passing nothing True is assumed.
                # i.e. the only way to disable somthing that defaults to True is to use the long form:
                # "--a_default_true_arg False" becomes False, while "--a_default_false_arg" becomes None,
                # which then becomes True here.

                v = True

        modified_args[k] = v
    return Namespace(**modified_args)


def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
    r"""Scans the Trainer signature and returns argument names, types and default values.

    Returns:
        List with tuples of 3 values:
        (argument name, set with argument types, argument default value).

    Examples:
        >>> from pytorch_lightning import Trainer
        >>> args = get_init_arguments_and_types(Trainer)
        >>> import pprint
        >>> pprint.pprint(sorted(args))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [('accumulate_grad_batches',
          (<class 'int'>, typing.Dict[int, int], typing.List[list]),
          1),
         ...
         ('callbacks',
          (typing.List[pytorch_lightning.callbacks.base.Callback],
           <class 'NoneType'>),
           None),
         ('check_val_every_n_epoch', (<class 'int'>,), 1),
         ...
         ('max_epochs', (<class 'int'>,), 1000),
         ...
         ('precision', (<class 'int'>,), 32),
         ('prepare_data_per_node', (<class 'bool'>,), True),
         ('process_position', (<class 'int'>,), 0),
         ('profiler',
          (<class 'pytorch_lightning.profiler.profilers.BaseProfiler'>,
           <class 'bool'>,
           <class 'NoneType'>),
          None),
         ...
    """
    trainer_default_params = inspect.signature(cls).parameters
    name_type_default = []
    for arg in trainer_default_params:
        arg_type = trainer_default_params[arg].annotation
        arg_default = trainer_default_params[arg].default
        try:
            arg_types = tuple(arg_type.__args__)
        except AttributeError:
            arg_types = (arg_type,)

        name_type_default.append((arg, arg_types, arg_default))

    return name_type_default


def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
    r"""Extends existing argparse by default `Trainer` attributes.

    Args:
        parent_parser:
            The custom cli arguments parser, which will be extended by
            the Trainer default arguments.

    Only arguments of the allowed types (str, float, int, bool) will
    extend the `parent_parser`.

    Examples:
        >>> import argparse
        >>> import pprint
        >>> from pytorch_lightning import Trainer
        >>> parser = argparse.ArgumentParser()
        >>> parser = Trainer.add_argparse_args(parser)
        >>> args = parser.parse_args([])
        >>> pprint.pprint(vars(args))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {...
         'check_val_every_n_epoch': 1,
         'checkpoint_callback': True,
         'default_root_dir': None,
         'deterministic': False,
         'distributed_backend': None,
         'early_stop_callback': False,
         ...
         'logger': True,
         'max_epochs': 1000,
         'max_steps': None,
         'min_epochs': 1,
         'min_steps': None,
         ...
         'profiler': None,
         'progress_bar_refresh_rate': 1,
         ...}

    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False,)

    blacklist = ['kwargs']
    depr_arg_names = cls.get_deprecated_arg_names() + blacklist

    allowed_types = (str, int, float, bool)

    # TODO: get "help" from docstring :)
    for arg, arg_types, arg_default in (
        at for at in get_init_arguments_and_types(cls) if at[0] not in depr_arg_names
    ):
        arg_types = [at for at in allowed_types if at in arg_types]
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type = parsing.str_to_bool
            # if only two args (str, bool)
            elif len(arg_types) == 2 and set(arg_types) == {str, bool}:
                use_type = parsing.str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == 'gpus' or arg == 'tpu_cores':
            use_type = _gpus_allowed_type
            arg_default = _gpus_arg_default

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == 'track_grad_norm':
            use_type = float

        parser.add_argument(
            f'--{arg}',
            dest=arg,
            default=arg_default,
            type=use_type,
            help='autogenerated by pl.Trainer',
            **arg_kwargs,
        )

    return parser


def _gpus_allowed_type(x) -> Union[int, str]:
    if ',' in x:
        return str(x)
    else:
        return int(x)


def _gpus_arg_default(x) -> Union[int, str]:
    if ',' in x:
        return str(x)
    else:
        return int(x)


def _int_or_float_type(x) -> Union[int, float]:
    if '.' in str(x):
        return float(x)
    else:
        return int(x)
