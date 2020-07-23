import pytest
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

from pytorch_lightning.metrics.functional.nlp import bleu_score

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.sentence_bleu
HYPOTHESIS1 = tuple(
    "It is a guide to action which ensures that the military always obeys the commands of the party".split()
)
REFERENCE1 = tuple("It is a guide to action that ensures that the military will forever heed Party commands".split())
REFERENCE2 = tuple(
    "It is a guiding principle which makes the military forces always being under the command of the Party".split()
)
REFERENCE3 = tuple("It is the practical guide for the army always to heed the directions of the party".split())


# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu
HYP1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split()
HYP2 = "he read the book because he was interested in world history".split()

REF1A = "It is a guide to action that ensures that the military will forever heed Party commands".split()
REF1B = "It is a guiding principle which makes the military force always being under the command of the Party".split()
REF1C = "It is the practical guide for the army always to heed the directions of the party".split()
REF2A = "he was interested in world history because he read the book".split()

LIST_OF_REFERENCES = [[REF1A, REF1B, REF1C], [REF2A]]
HYPOTHESES = [HYP1, HYP2]

# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.SmoothingFunction
smooth_func = SmoothingFunction().method2


@pytest.mark.parametrize(
    ["weights", "n_gram", "smooth_func", "smooth"],
    [
        pytest.param([1], 1, None, False),
        pytest.param([0.5, 0.5], 2, smooth_func, True),
        pytest.param([0.333333, 0.333333, 0.333333], 3, None, False),
        pytest.param([0.25, 0.25, 0.25, 0.25], 4, smooth_func, True),
    ],
)
def test_bleu_score(weights, n_gram, smooth_func, smooth):
    nltk_output = sentence_bleu(
        [REFERENCE1, REFERENCE2, REFERENCE3], HYPOTHESIS1, weights=weights, smoothing_function=smooth_func
    )
    pl_output = bleu_score([HYPOTHESIS1], [[REFERENCE1, REFERENCE2, REFERENCE3]], n_gram=n_gram, smooth=smooth)
    assert torch.allclose(pl_output, torch.tensor(nltk_output))

    nltk_output = corpus_bleu(LIST_OF_REFERENCES, HYPOTHESES, weights=weights, smoothing_function=smooth_func)
    pl_output = bleu_score(HYPOTHESES, LIST_OF_REFERENCES, n_gram=n_gram, smooth=smooth)
    assert torch.allclose(pl_output, torch.tensor(nltk_output))


def test_bleu_empty():
    hyp = [[]]
    ref = [[[]]]
    assert bleu_score(hyp, ref) == torch.tensor(0.0)


def test_no_4_gram():
    hyps = [["My", "full", "pytorch-lightning"]]
    refs = [[["My", "full", "pytorch-lightning", "test"], ["Completely", "Different"]]]
    assert bleu_score(hyps, refs) == torch.tensor(0.0)
