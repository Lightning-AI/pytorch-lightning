import torch
from typing import List
from pytorch_lightning.metrics import Metric

from pytorch_lightning.metrics.utils import get_mini_groups


IGNORE_IDX = -100

class RetrievalMetric(Metric):
    """
    Compute a metric for Information Retrieval by grouping predictions on the same
    document using indexes. Detailed information about metrics are contained
    in sub-classes.

    It may be possible that a document has no positive label: this case can
    be managed in different ways using the `empty_documents` parameter:
    - `skip`: those documents are skipped (default)
    - `error`: a `ValueError` is raised
    - `positive`: those documents are counted as positive predictions (1.0)
    - `negative`: those documents are counted as negative predictions (0.0)

    Entries with targets equal to `exclude` will be ignored.
    Subclasses must override at least the `metric` method.
    """

    options = ['error', 'skip', 'positive', 'negative']

    def __init__(self, dist_sync_on_step: bool = False, empty_documents: str = 'skip', exclude: int = IGNORE_IDX):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if empty_documents not in self.options:
            raise ValueError(
                f"`empty_documents` received a wrong value {empty_documents}."
                f"Allowed values are {self.options}"
            )

        self.empty_documents = empty_documents
        self.exclude = exclude

        self.add_state("idx", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")
        self.add_state("preds", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")

    def update(self, idx: torch.Tensor, preds: torch.Tensor, target: torch.Tensor):
        assert idx.shape == preds.shape == target.shape, (
            f"Indexes, predicions and targets must be of the same shape"
        )

        self.idx = torch.cat([self.idx, idx])
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])
    
    def compute(self):
        res = []
        for group in get_mini_groups(self.idx):
            if self.target[group].sum() == 0:
                if self.empty_documents == 'error':
                    raise ValueError(
                        f"{self.__class__.__name__} was provided with a prediction with no positive values, idx: {group}"
                    )
                elif self.empty_documents == 'positive': res.append(torch.tensor([1.0]))
                elif self.empty_documents == 'negative': res.append(torch.tensor([0.0]))
            else:
                res.append(
                    self.metric(group)
                )
        return torch.stack(res).mean()

    def metric(self, group: List[int]):
        r""" Compute a metric over a single group. """
        raise NotImplementedError("This method must be overridden by subclasses")
