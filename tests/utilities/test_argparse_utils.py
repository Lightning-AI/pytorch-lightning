from pytorch_lightning.utilities.argparse_utils import parse_args_from_docstring


def test_parse_args_from_docstring_normal():
    args_help = parse_args_from_docstring(
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
    args_help = parse_args_from_docstring(
        """Constrain image dataset

        Args:

        Returns:

        Examples:
        """
    )
    assert len(args_help.keys()) == 0
