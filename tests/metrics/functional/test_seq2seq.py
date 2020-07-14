import pytest
import torch

from pytorch_lightning.metrics.functional.seq2seq import (
    bleu
)

@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([['My', 'full', 'pl', 'test'], ['Another', 'Sentence']],[[['My', 'full', 'pl', 'test'], ['Completely', 'Different']], [['No', 'Match']]], 0.8409)
])
def test_mse(pred, target, expected):
    score = bleu(torch.tensor(pred), torch.tensor(target))
    assert pytest.approx(score.item(),0.1) == expected