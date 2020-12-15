import math
import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score as sk_average_precision

from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.retrieval.mean_average_precision import RetrievalMAP


@pytest.mark.parametrize(['sklearn_metric', 'torch_class_metric'],[
    [sk_average_precision, RetrievalMAP],
])
def test_against_sklearn(sklearn_metric, torch_class_metric):
    """Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    rounds = 20
    sizes = [1, 4, 10, 100]
    batch_sizes = [1, 4, 10]
    query_without_relevant_docs_options = ['skip', 'pos', 'neg']

    for batch_size in batch_sizes:
        for size in sizes:
            for _ in range(rounds):
                for behaviour in query_without_relevant_docs_options:

                    metric = torch_class_metric(query_without_relevant_docs=behaviour)
                    shape = (size,)

                    indexes = []
                    preds = []
                    target = []

                    for i in range(batch_size):
                        indexes.append(
                            np.ones(shape, dtype=int) * i
                        )
                        preds.append(
                            np.random.randn(*shape)
                        )
                        target.append(
                            np.random.randn(*shape) > 0
                        )

                    # compute sk metric with multiple iterations using the base `sklearn_metric`
                    sk_results = []
                    for b, a in zip(target, preds):
                        res = sklearn_metric(b, a)

                        if math.isnan(res):
                            if behaviour == 'skip':
                                pass
                            elif behaviour == 'pos':
                                sk_results.append(
                                    torch.tensor(1.0, device=device)
                                )
                            else:
                                sk_results.append(
                                    torch.tensor(0.0, device=device)
                                )
                        else:
                            sk_results.append(
                                torch.tensor(res, device=device)
                            )
                    if len(sk_results) > 0:
                        sk_results = torch.stack(sk_results).mean()
                    else:
                        sk_results = torch.tensor(0.0, device=device)

                    indexes = torch.cat([torch.tensor(i) for i in indexes])
                    preds = torch.cat([torch.tensor(p) for p in preds])
                    target = torch.cat([torch.tensor(t) for t in target])

                    perm = torch.randperm(indexes.nelement())
                    indexes = indexes.view(-1)[perm].view(indexes.size())
                    preds = preds.view(-1)[perm].view(preds.size())
                    target = target.view(-1)[perm].view(target.size())

                    # shuffle ids to require also sorting of documents ability from the lightning metric
                    pl_result = metric(indexes, preds, target)

                    assert torch.allclose(sk_results.float(), pl_result.float(), equal_nan=True)

    # check error when `query_without_relevant_docs='error'` is raised correctly
    indexes = torch.tensor([0, 0, 0], device=device, dtype=torch.int64)
    preds = torch.tensor([0.1, 0.2, 0.3], device=device, dtype=torch.float32)
    target = torch.tensor([False, False, False], device=device, dtype=torch.bool)

    metric = torch_class_metric(query_without_relevant_docs='error')

    try:
        metric(indexes, preds, target)
    except Exception as e:
        assert isinstance(e, ValueError)

    # check ValueError with non-accepted argument
    try:
        metric = torch_class_metric(query_without_relevant_docs='casual_argument')
    except Exception as e:
        assert isinstance(e, ValueError)
