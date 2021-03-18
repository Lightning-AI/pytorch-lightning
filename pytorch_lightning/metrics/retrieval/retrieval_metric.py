from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric

from pytorch_lightning.metrics.utils import get_group_indexes

#: get_group_indexes is used to group predictions belonging to the same query

IGNORE_IDX = -100


class RetrievalMetric(Metric, ABC):
    r"""
    Works with binary data. Accepts integer or float predictions from a model output.

    Forward accepts
    - ``indexes`` (long tensor): ``(N, ...)``
    - ``preds`` (float or int tensor): ``(N, ...)``
    - ``target`` (long or bool tensor): ``(N, ...)``

    `indexes`, `preds` and `target` must have the same dimension and will be flatten
    to single dimension once provided.

    `indexes` indicate to which query a prediction belongs.
    Predictions will be first grouped by indexes. Then the
    real metric, defined by overriding the `_metric` method,
    will be computed as the mean of the scores over each query.

    Args:
        query_without_relevant_docs:
            Specify what to do with queries that do not have at least a positive target. Choose from:

            - ``'skip'``: skip those queries (default); if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``
            - ``'pos'``: score on those queries is counted as ``1.0``
            - ``'neg'``: score on those queries is counted as ``0.0``
        exclude:
            Do not take into account predictions where the target is equal to this value. default `-100`
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects
            the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    """

    def __init__(
        self,
        query_without_relevant_docs: str = 'skip',
        exclude: int = IGNORE_IDX,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )

        query_without_relevant_docs_options = ('error', 'skip', 'pos', 'neg')
        if query_without_relevant_docs not in query_without_relevant_docs_options:
            raise ValueError(
                f"`query_without_relevant_docs` received a wrong value {query_without_relevant_docs}. "
                f"Allowed values are {query_without_relevant_docs_options}"
            )

        self.query_without_relevant_docs = query_without_relevant_docs
        self.exclude = exclude

        self.add_state("idx", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, idx: torch.Tensor, preds: torch.Tensor, target: torch.Tensor) -> None:
        if not (idx.shape == target.shape == preds.shape):
            raise ValueError("`idx`, `preds` and `target` must be of the same shape")

        idx = idx.to(dtype=torch.int64).flatten()
        preds = preds.to(dtype=torch.float32).flatten()
        target = target.to(dtype=torch.int64).flatten()

        self.idx.append(idx)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        r"""
        First concat state `idx`, `preds` and `target` since they were stored as lists. After that,
        compute list of groups that will help in keeping together predictions about the same query.
        Finally, for each group compute the `_metric` if the number of positive targets is at least
        1, otherwise behave as specified by `self.query_without_relevant_docs`.
        """

        idx = torch.cat(self.idx, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        res = []
        kwargs = {'device': idx.device, 'dtype': torch.float32}

        groups = get_group_indexes(idx)
        for group in groups:

            mini_preds = preds[group]
            mini_target = target[group]

            if not mini_target.sum():
                if self.query_without_relevant_docs == 'error':
                    raise ValueError(
                        f"`{self.__class__.__name__}.compute()` was provided with "
                        f"a query without positive targets, indexes: {group}"
                    )
                if self.query_without_relevant_docs == 'pos':
                    res.append(torch.tensor(1.0, **kwargs))
                elif self.query_without_relevant_docs == 'neg':
                    res.append(torch.tensor(0.0, **kwargs))
            else:
                res.append(self._metric(mini_preds, mini_target))

        if len(res) > 0:
            return torch.stack(res).mean()
        return torch.tensor(0.0, **kwargs)

    @abstractmethod
    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute a metric over a predictions and target of a single group.
        This method should be overridden by subclasses.
        """
