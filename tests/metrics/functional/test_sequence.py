import pytest
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

from pytorch_lightning.metrics.functional.sequence import bleu_score

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.sentence_bleu
hypothesis1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split(
    " "
)
hypothesis2 = "It is to insure the troops forever hearing the activity guidebook that party direct".split(" ")
reference1 = "It is a guide to action that ensures that the military will forever heed Party commands".split(" ")
reference2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party".split(
    " "
)
reference3 = "It is the practical guide for the army always to heed the directions of the party".split(" ")


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
    ["weights", "n", "smooth_func", "smooth"],
    [
        pytest.param((1, 0, 0, 0), 1, None, False),
        pytest.param((0.5, 0.5, 0, 0), 2, smooth_func, True),
        pytest.param((0.333333, 0.333333, 0.333333, 0), 3, None, False),
        pytest.param((0.25, 0.25, 0.25, 0.25), 4, smooth_func, True),
    ],
)
class TestBLEUScore:
    def test_with_sentence_bleu(self, weights, n, smooth_func, smooth):
        nltk_output = sentence_bleu(
            [reference1, reference2, reference3], hypothesis1, weights=weights, smoothing_function=smooth_func
        )
        pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=n, smooth=smooth)
        assert torch.allclose(pl_output, torch.tensor(nltk_output))

    def test_with_corpus_bleu(self, weights, n, smooth_func, smooth):
        nltk_output = corpus_bleu(list_of_references, hypotheses, weights=weights, smoothing_function=smooth_func)
        pl_output = bleu_score(hypotheses, list_of_references, n=n, smooth=smooth)
        assert torch.allclose(pl_output, torch.tensor(nltk_output))


def test_bleu_empty():
    hyp = [[]]
    ref = [[[]]]
    assert bleu_score(hyp, ref) == torch.tensor(0.0)


def test_no_4_gram():
    hyps = [["My", "full", "pytorch-lightning"]]
    refs = [[["My", "full", "pytorch-lightning", "test"], ["Completely", "Different"]]]
    assert bleu_score(hyps, refs) == torch.tensor(0.0)
