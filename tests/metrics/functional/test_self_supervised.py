import pytest
import torch
from sklearn.metrics import pairwise

from pytorch_lightning.metrics.functional.self_supervised import embedding_similarity


@pytest.mark.parametrize('similarity', ['cosine', 'dot'])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_against_sklearn(similarity, reduction):
    """Compare PL metrics to sklearn version."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch = torch.randn(5, 10, device=device)  # 100 samples in 10 dimensions

    pl_dist = embedding_similarity(batch, similarity=similarity, reduction=reduction, zero_diagonal=False)

    def sklearn_embedding_distance(batch, similarity, reduction):

        metric_func = {'cosine': pairwise.cosine_similarity, 'dot': pairwise.linear_kernel}[similarity]

        dist = metric_func(batch, batch)
        if reduction == 'mean':
            return dist.mean(axis=-1)
        if reduction == 'sum':
            return dist.sum(axis=-1)
        return dist

    sk_dist = sklearn_embedding_distance(batch.cpu().detach().numpy(), similarity=similarity, reduction=reduction)
    sk_dist = torch.tensor(sk_dist, dtype=torch.float, device=device)

    assert torch.allclose(sk_dist, pl_dist)
