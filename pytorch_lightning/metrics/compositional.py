from typing import Callable, Union

import torch

from pytorch_lightning.metrics.metric import Metric


class CompositionalMetric(Metric):
    """Composition of two metrics with a specific operator
    which will be executed upon metric's compute

    """

    def __init__(
        self,
        operator: Callable,
        metric_a: Union[Metric, int, float, torch.Tensor],
        metric_b: Union[Metric, int, float, torch.Tensor, None],
    ):
        """

        Args:
            operator: the operator taking in one (if metric_b is None)
                or two arguments. Will be applied to outputs of metric_a.compute()
                and (optionally if metric_b is not None) metric_b.compute()
            metric_a: first metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None
        """
        super().__init__()

        self.op = operator

        if isinstance(metric_a, torch.Tensor):
            self.register_buffer("metric_a", metric_a)
        else:
            self.metric_a = metric_a

        if isinstance(metric_b, torch.Tensor):
            self.register_buffer("metric_b", metric_b)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn=None):
        # No syncing required here. syncing will be done in metric_a and metric_b
        pass

    def update(self, *args, **kwargs):
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **self.metric_a._filter_kwargs(**kwargs))

        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **self.metric_b._filter_kwargs(**kwargs))

    def compute(self):

        # also some parsing for kwargs?
        if isinstance(self.metric_a, Metric):
            val_a = self.metric_a.compute()
        else:
            val_a = self.metric_a

        if isinstance(self.metric_b, Metric):
            val_b = self.metric_b.compute()
        else:
            val_b = self.metric_b

        if val_b is None:
            return self.op(val_a)

        return self.op(val_a, val_b)

    def reset(self):
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False):
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self):
        repr_str = (
            self.__class__.__name__
            + f"(\n  {self.op.__name__}(\n    {repr(self.metric_a)},\n    {repr(self.metric_b)}\n  )\n)"
        )

        return repr_str
