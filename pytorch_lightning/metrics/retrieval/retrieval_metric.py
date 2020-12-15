from abc import ABC, abstractmethod
import torch
from typing import List, Optional, Callable, Any

from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.utils import get_mini_groups
#: get_mini_groups is used to group predictions belonging to the same query

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
        query_without_relevant_docs:
            Specify what to do with queries that do not have at least a positive target. Choose from:

            - ``'skip'``: skip those queries (default); if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``
            - ``'pos'``: score on those queries is counted as ``1.0``
            - ``'neg'``: score on those queries is counted as ``0.0``

        exclude:
            Do not take into account predictions where the target is equal to this value. default: -100
    """

    query_without_relevant_docs_options = ['error', 'skip', 'pos', 'neg']

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

        if query_without_relevant_docs not in self.query_without_relevant_docs_options:
            raise ValueError(
                f"`query_without_relevant_docs` received a wrong value {query_without_relevant_docs}. "
                f"Allowed values are {self.query_without_relevant_docs_options}"
            )

        self.query_without_relevant_docs = query_without_relevant_docs
        self.exclude = exclude

        self.add_state("idx", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")
        self.add_state("preds", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")

    def update(self, idx: torch.Tensor, preds: torch.Tensor, target: torch.Tensor) -> None:
        if not (idx.shape == target.shape == preds.shape):
            raise ValueError(
                "Indexes, preds and targets must be of the same shape after preds normalization"
            )

        idx = idx.to(dtype=torch.int64).flatten()
        preds = preds.to(dtype=torch.float32).flatten()
        target = target.to(dtype=torch.int64).flatten()

        self.idx = torch.cat([self.idx, idx])
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self) -> torch.Tensor:
        res = []
        device = self.idx.device

        for group in get_mini_groups(self.idx):
            if self.target[group].sum() == 0:
                if self.query_without_relevant_docs == 'error':
                    raise ValueError(
                        f"`{self.__class__.__name__}.compute()` was provided with "
                        f"a query without positive targets, indexes: {group}"
                    )
                if self.query_without_relevant_docs == 'pos':
                    res.append(torch.tensor(1.0, device=device))
                elif self.query_without_relevant_docs == 'neg':
                    res.append(torch.tensor(0.0, device=device))
            else:
                res.append(
                    self._metric(group)
                )
        if len(res) > 0:
            return torch.stack(res).mean()
        return torch.tensor(0.0, device=device)

    @abstractmethod
    def _metric(self, group: List[int]) -> torch.Tensor:
        r"""
        Compute a metric over a single group.
        This method must be overridden by subclasses.
        """
