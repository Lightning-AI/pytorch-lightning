from typing import List

from pytorch_lightning.metrics.retrieval.retrieval_metric import RetrievalMetric
from pytorch_lightning.metrics.functional.ir_precision import precision


class PrecisionAtK(RetrievalMetric):
    """
    Precision at K computes the P@K over multiple retrieved documents for each query.
    Each precision at k computation on each query can be done on a different number
    of predictions thanks to the usage of a tensor dedicated to separate query results.

    Example:

        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = torch.tensor([False, False, True, False, True, False, False])

        >>> p_k = PrecisionAtK(k=1)
        >>> p_k(indexes, preds, target)
        >>> p_k.compute()
        ... 0.5
    """

    def __init__(self, *args, k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def metric(self, group: List[int]):
        _preds = self.preds[group]
        _target = self.target[group]
        valid_indexes = (_target != self.exclude)
        return precision(_preds[valid_indexes], _target[valid_indexes], k=self.k)
