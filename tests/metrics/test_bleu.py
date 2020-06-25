import pytest

from pytorch_lightning.metrics.bleu import bleu_score, best_match_len


def test_best_match_len():
    tl = "the FAST brown fox jumped over the lazy dog".split()
    r1 = ["the quick brown fox jumped over the lazy dog"]
    r2 = ["the quick brown fox jumped over the the lazy cat"]
    ref_len_1, trans_len_1 = best_match_len(tl, r1)
    assert ref_len_1 == trans_len_1

    ref_len_2, trans_len_2 = best_match_len(tl, r2)
    assert ref_len_2 != trans_len_2


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bleu_score(n):
    t = "the FAST brown fox jumped over the lazy dog"
    r = ["the quick brown fox jumped over the lazy dog", "the quick brown fox jumped over the the lazy cat"]
    bleu = bleu_score(t, r, n)
    assert isinstance(bleu, float)
