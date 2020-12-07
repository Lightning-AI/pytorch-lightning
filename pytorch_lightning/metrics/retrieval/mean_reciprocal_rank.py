from typing import List

from pytorch_lightning.metrics.retrieval.retrieval_metric import RetrievalMetric
from pytorch_lightning.metrics.functional.ir_reciprocal_rank import retrieval_reciprocal_rank


class RetrievalMRR(RetrievalMetric):
    """
    Mean Reciprocal Rank computes the MRR over multiple retrieved documents for each query.
    Each reciprocal rank computation on a single query can be done on a different number of
    predictions thanks to the usage of a tensor dedicated to separate query results.

    Example:

        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = torch.tensor([False, False, True, False, True, False, False])

        >>> mrr = RetrievalMRR()
        >>> mrr(indexes, preds, target)
        >>> mrr.compute()
        ... 0.75
    """

    def metric(self, group: List[int]):
        _preds = self.preds[group]
        _target = self.target[group]
        valid_indexes = (_target != self.exclude)
        return retrieval_reciprocal_rank(_preds[valid_indexes], _target[valid_indexes])
