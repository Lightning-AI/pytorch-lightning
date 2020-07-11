import pytest
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from pytorch_lightning.metrics.sequence import BLEUScore

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu
hyp1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split(" ")
hyp2 = "he read the book because he was interested in world history".split(" ")

ref1a = "It is a guide to action that ensures that the military will forever heed Party commands".split(" ")
ref1b = "It is the guiding principle which guarantees the military forces always being under the command of the Party".split(
    " "
)
ref1c = "It is the practical guide for the army always to heed the directions of the party".split(" ")
ref2a = "he was interested in world history because he read the book".split(" ")

list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
hypotheses = [hyp1, hyp2]

# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.SmoothingFunction
smooth_func = SmoothingFunction().method2


@pytest.mark.parametrize(
    ["weights", "n_gram", "smooth_func", "smooth"],
    [
        pytest.param((1, 0, 0, 0), 1, None, False),
        pytest.param((0.5, 0.5, 0, 0), 2, smooth_func, True),
        pytest.param((0.333333, 0.333333, 0.333333, 0), 3, None, False),
        pytest.param((0.25, 0.25, 0.25, 0.25), 4, smooth_func, True),
    ],
)
def test_bleu(weights, n_gram, smooth_func, smooth):
    bleu = BLEUScore(n_gram=n_gram, smooth=smooth)
    # assert bleu.name == "bleu"

    nltk_output = corpus_bleu(list_of_references, hypotheses, weights=weights, smoothing_function=smooth_func)
    pl_output = bleu(hypotheses, list_of_references)
    assert torch.allclose(pl_output, torch.tensor(nltk_output))
    assert isinstance(pl_output, torch.Tensor)
