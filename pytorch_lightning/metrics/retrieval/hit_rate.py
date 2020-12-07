from typing import List, Optional, Callable, Any

from pytorch_lightning.metrics.retrieval.retrieval_metric import RetrievalMetric, IGNORE_IDX
from pytorch_lightning.metrics.functional.hit_rate import hit_rate


class HitRate(RetrievalMetric):
    """
    Hit Rate at K computes the HR@K over multiple retrieved documents for each query.
    Each hit rate at k computation over a single query can be done on a different number 
    of predictions thanks to the usage of a tensor dedicated to separate query results.

    Notice that HR@1 == P@1

    Example:

        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = torch.tensor([False, False, True, False, True, False, False])

        >>> hr_k = HitRate(k=1)
        >>> hr_k(indexes, preds, target)
        >>> hr_k.compute()
        ... 0.5
    """

    def __init__(self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        empty_documents: str = 'skip',
        exclude: int = IGNORE_IDX,
        k: int = 1
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            empty_documents=empty_documents,
            exclude=exclude,
        )
        self.k = k

    def metric(self, group: List[int]):
        _preds = self.preds[group]
        _target = self.target[group]
        valid_indexes = (_target != self.exclude)
        return hit_rate(_preds[valid_indexes], _target[valid_indexes], k=self.k)