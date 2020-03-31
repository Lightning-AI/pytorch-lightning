import numbers
from typing import Any, Union

import numpy as np

import torch
from torch.utils.data._utils.collate import default_convert

from pytorch_lightning import _logger as lightning_logger
from pytorch_lightning.metrics.metric import BaseMetric
from pytorch_lightning.utilities.apply_to_collection import apply_to_collection


class SklearnMetric(BaseMetric):
    def __init__(self, metric_name: str,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM, **kwargs):
        """
        Bridge between PyTorch Lightning and scikit-learn metrics

        .. warning::
            Every metric call will cause a GPU synchronization, which may slow down your code

        Args:
            metric_name: the metric name to import anc compute from scikit-learn.metrics
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
            **kwargs: additonal keyword arguments (will be forwarded to metric call)
        """
        super().__init__(name=metric_name, reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.metric_kwargs = kwargs

        lightning_logger.debug(
            'Every metric call will cause a GPU synchronization, which may slow down your code')

    @property
    def metric_fn(self):
        import sklearn.metrics
        return getattr(sklearn.metrics, self.name)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Carries the actual metric computation and therefore co
        Args:
            *args: Positional arguments forwarded to metric call
            **kwargs: keyword arguments forwarded to metric call

        Returns:
            the metric value

        """
        # we need to include numpy arrays here, since otherwise they will also be treated as sequences
        args = apply_to_collection(args, (torch.Tensor, np.ndarray), _convert_to_numpy)
        kwargs = apply_to_collection(kwargs, (torch.Tensor, np.ndarray), _convert_to_numpy)

        return _convert_to_tensor(self.metric_fn(*args, **kwargs, **self.metric_kwargs))


def _convert_to_tensor(data: Any) -> Any:
    """
    Maps all kind of collections and numbers to tensors

    Args:
        data: the data to convert to tensor

    Returns:
        the converted data

    """
    if isinstance(data, numbers.Number):
        return torch.tensor([data])

    else:
        return default_convert(data)


def _convert_to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    converts all tensors and numpy arrays to numpy arrays
    Args:
        data: the tensor or array to convert to numpy

    Returns:
        the resulting numpy array

    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    return data
