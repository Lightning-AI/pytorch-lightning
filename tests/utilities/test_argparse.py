import io
from argparse import ArgumentParser, Namespace
from typing import List
from unittest.mock import MagicMock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.argparse import (
    _get_abbrev_qualified_cls_name,
    _gpus_allowed_type,
    _int_or_float_type,
    _parse_args_from_docstring,
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
)


class ArgparseExample:

    def __init__(self, a: int = 0, b: str = '', c: bool = False):
        self.a = a
        self.b = b
        self.c = c


def test_from_argparse_args():
    args = Namespace(a=1, b='test', c=True, d='not valid')
    my_instance = from_argparse_args(ArgparseExample, args)
    assert my_instance.a == 1
    assert my_instance.b == 'test'
    assert my_instance.c

    parser = ArgumentParser()
    mock_trainer = MagicMock()
    _ = from_argparse_args(mock_trainer, parser)
    mock_trainer.parse_argparser.assert_called_once_with(parser)


def test_parse_argparser():
    args = Namespace(a=1, b='test', c=None, d='not valid')
    new_args = parse_argparser(ArgparseExample, args)
    assert new_args.a == 1
    assert new_args.b == 'test'
    assert new_args.c
    assert new_args.d == 'not valid'


def test_parse_args_from_docstring_normal():
    args_help = _parse_args_from_docstring(
        """Constrain image dataset

        Args:
            root: Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            train: If ``True``, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            normalize: mean and std deviation of the MNIST dataset.
            download: If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            num_samples: number of examples per selected class/digit
            digits: list selected MNIST digits/classes

        Examples:
            >>> dataset = TrialMNIST(download=True)
            >>> len(dataset)
            300
            >>> sorted(set([d.item() for d in dataset.targets]))
            [0, 1, 2]
            >>> torch.bincount(dataset.targets)
            tensor([100, 100, 100])
        """
    )

    expected_args = ['root', 'train', 'normalize', 'download', 'num_samples', 'digits']
    assert len(args_help.keys()) == len(expected_args)
    assert all([x == y for x, y in zip(args_help.keys(), expected_args)])
    assert args_help['root'] == 'Root directory of dataset where ``MNIST/processed/training.pt``' \
                                ' and  ``MNIST/processed/test.pt`` exist.'
    assert args_help['normalize'] == 'mean and std deviation of the MNIST dataset.'


def test_parse_args_from_docstring_empty():
    args_help = _parse_args_from_docstring(
        """Constrain image dataset

        Args:

        Returns:

        Examples:
        """
    )
    assert len(args_help.keys()) == 0


def test_get_abbrev_qualified_cls_name():
    assert _get_abbrev_qualified_cls_name(Trainer) == "pl.Trainer"

    class NestedClass:
        pass

    assert not __name__.startswith("pytorch_lightning.")
    expected_name = f"{__name__}.test_get_abbrev_qualified_cls_name.<locals>.NestedClass"
    assert _get_abbrev_qualified_cls_name(NestedClass) == expected_name


class AddArgparseArgsExampleClass:
    """
    Args:
        my_parameter: A thing.
    """

    def __init__(self, my_parameter: int = 0):
        pass

    @staticmethod
    def get_deprecated_arg_names() -> List[str]:
        return []


class AddArgparseArgsExampleClassViaInit:

    def __init__(self, my_parameter: int = 0):
        """
        Args:
            my_parameter: A thing.
        """
        pass


class AddArgparseArgsExampleClassNoDoc:

    def __init__(self, my_parameter: int = 0):
        pass


def extract_help_text(parser):
    help_str_buffer = io.StringIO()
    parser.print_help(file=help_str_buffer)
    help_str_buffer.seek(0)
    return help_str_buffer.read()


@pytest.mark.parametrize(["cls", "name"], [
    [AddArgparseArgsExampleClass, "AddArgparseArgsExampleClass"],
    [AddArgparseArgsExampleClassViaInit, "AddArgparseArgsExampleClassViaInit"],
    [AddArgparseArgsExampleClassNoDoc, "AddArgparseArgsExampleClassNoDoc"],
])
def test_add_argparse_args(cls, name):
    """
    Tests that ``add_argparse_args`` handles argument groups correctly, and
    can be parsed.
    """
    parser = ArgumentParser()
    parser_main = parser.add_argument_group("main")
    parser_main.add_argument("--main_arg", type=str, default="")
    parser_old = parser  # For testing.
    parser = add_argparse_args(cls, parser)
    assert parser is parser_old

    # Check nominal argument groups.
    help_text = extract_help_text(parser)
    assert "main:" in help_text
    assert "--main_arg" in help_text
    assert f"{name}:" in help_text
    assert "--my_parameter" in help_text
    if cls is not AddArgparseArgsExampleClassNoDoc:
        assert "A thing" in help_text

    fake_argv = ["--main_arg=abc", "--my_parameter=2"]
    args = parser.parse_args(fake_argv)
    assert args.main_arg == "abc"
    assert args.my_parameter == 2


def test_negative_add_argparse_args():
    with pytest.raises(RuntimeError, match="Please only pass an ArgumentParser instance."):
        parser = ArgumentParser()
        add_argparse_args(AddArgparseArgsExampleClass, parser.add_argument_group("bad workflow"))


def test_add_argparse_args_no_argument_group():
    """
    Tests that ``add_argparse_args(..., use_argument_group=False)`` (old
    workflow) handles argument groups correctly, and can be parsed.
    """
    parser = ArgumentParser()
    parser.add_argument("--main_arg", type=str, default="")
    parser_old = parser  # For testing.
    parser = add_argparse_args(AddArgparseArgsExampleClass, parser, use_argument_group=False)
    assert parser is not parser_old

    # Check arguments.
    help_text = extract_help_text(parser)
    assert "--main_arg" in help_text
    assert "--my_parameter" in help_text
    assert "AddArgparseArgsExampleClass:" not in help_text

    fake_argv = ["--main_arg=abc", "--my_parameter=2"]
    args = parser.parse_args(fake_argv)
    assert args.main_arg == "abc"
    assert args.my_parameter == 2


def test_gpus_allowed_type():
    assert _gpus_allowed_type('1,2') == '1,2'
    assert _gpus_allowed_type('1') == 1


def test_int_or_float_type():
    assert isinstance(_int_or_float_type('0.0'), float)
    assert isinstance(_int_or_float_type('0'), int)
