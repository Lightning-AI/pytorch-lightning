import pytest
import torch

from pytorch_lightning.metrics.nlp import BLEUScore

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu
HYP1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split()
HYP2 = "he read the book because he was interested in world history".split()

REF1A = "It is a guide to action that ensures that the military will forever heed Party commands".split()
REF1B = "It is a guiding principle which makes the military forces always being under the command of the Party".split()
REF1C = "It is the practical guide for the army always to heed the directions of the party".split()
REF2A = "he was interested in world history because he read the book".split()

LIST_OF_REFERENCES = [[REF1A, REF1B, REF1C], [REF2A]]
HYPOTHESES = [HYP1, HYP2]


@pytest.mark.parametrize(
    ["n_gram", "smooth"],
    [pytest.param(1, True), pytest.param(2, False), pytest.param(3, True), pytest.param(4, False),],
)
def test_bleu(smooth, n_gram):
    bleu = BLEUScore(n_gram=n_gram, smooth=smooth)
    assert bleu.name == "bleu"

    pl_output = bleu(HYPOTHESES, LIST_OF_REFERENCES)
    assert isinstance(pl_output, torch.Tensor)
