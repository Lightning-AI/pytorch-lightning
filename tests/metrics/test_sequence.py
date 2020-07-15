import pytest
import torch

from pytorch_lightning.metrics.sequence import BLEUScore

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu
hyp1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split()
hyp2 = "he read the book because he was interested in world history".split()

ref1a = "It is a guide to action that ensures that the military will forever heed Party commands".split()
ref1b = "It is a guiding principle which makes the military forces always being under the command of the Party".split()
ref1c = "It is the practical guide for the army always to heed the directions of the party".split()
ref2a = "he was interested in world history because he read the book".split()

list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
hypotheses = [hyp1, hyp2]


@pytest.mark.parametrize(
    ["n_gram", "weights"],
    [
        pytest.param(1, [1]),
        pytest.param(2, [0.5, 0.5]),
        pytest.param(3, [0.333333, 0.333333, 0.333333]),
        pytest.param(4, [0.25, 0.25, 0.25, 0.25]),
    ],
)
def test_bleu(weights, n_gram):
    bleu = BLEUScore(n_gram=n_gram, weights=weights)
    assert bleu.name == "bleu"

    pl_output = bleu(hypotheses, list_of_references)
    assert isinstance(pl_output, float)
