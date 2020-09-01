# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Tuple, Union
from pytorch_lightning.utilities import argparse_utils
from argparse import ArgumentParser, Namespace


class ArgparseClass(object):

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""
        Scans the Trainer signature and returns argument names, types and default values.

        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).

        Examples:
            >>> args = Trainer.get_init_arguments_and_types()
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
        return argparse_utils.get_init_arguments_and_types(cls)

    @classmethod
    def get_deprecated_arg_names(cls) -> List:
        """Returns a list with deprecated Trainer arguments."""
        depr_arg_names = []
        for name, val in cls.__dict__.items():
            if name.startswith('DEPRECATED') and isinstance(val, (tuple, list)):
                depr_arg_names.extend(val)
        return depr_arg_names

    @classmethod
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
        parser = argparse_utils.add_argparse_args(cls, parent_parser)
        return parser

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        """Parse CLI arguments, required for custom bool types."""
        return argparse_utils.parse_argparser(cls, arg_parser)

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs) -> 'Trainer':
        """
        Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`Trainer`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid Trainer arguments.

        Example:
            >>> parser = ArgumentParser(add_help=False)
            >>> parser = Trainer.add_argparse_args(parser)
            >>> parser.add_argument('--my_custom_arg', default='something')  # doctest: +SKIP
            >>> args = Trainer.parse_argparser(parser.parse_args(""))
            >>> trainer = Trainer.from_argparse_args(args, logger=False)
        """
        return argparse_utils.from_argparse_args(cls, args)