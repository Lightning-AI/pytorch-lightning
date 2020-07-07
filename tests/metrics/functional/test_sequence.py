import pytest
from pytorch_lightning.metrics.functional.sequence import bleu_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.SmoothingFunction
smooth_func = SmoothingFunction().method2

# example taken from https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.sentence_bleu
hypothesis1 = [
    "It",
    "is",
    "a",
    "guide",
    "to",
    "action",
    "which",
    "ensures",
    "that",
    "the",
    "military",
    "always",
    "obeys",
    "the",
    "commands",
    "of",
    "the",
    "party",
]
hypothesis2 = [
    "It",
    "is",
    "to",
    "insure",
    "the",
    "troops",
    "forever",
    "hearing",
    "the",
    "activity",
    "guidebook",
    "that",
    "party",
    "direct",
]

reference1 = [
    "It",
    "is",
    "a",
    "guide",
    "to",
    "action",
    "that",
    "ensures",
    "that",
    "the",
    "military",
    "will",
    "forever",
    "heed",
    "Party",
    "commands",
]
reference2 = [
    "It",
    "is",
    "the",
    "guiding",
    "principle",
    "which",
    "guarantees",
    "the",
    "military",
    "forces",
    "always",
    "being",
    "under",
    "the",
    "command",
    "of",
    "the",
    "Party",
]
reference3 = [
    "It",
    "is",
    "the",
    "practical",
    "guide",
    "for",
    "the",
    "army",
    "always",
    "to",
    "heed",
    "the",
    "directions",
    "of",
    "the",
    "party",
]


def test_with_sentence_bleu():
    nltk_output = sentence_bleu([reference1, reference2, reference3], hypothesis1, weights=(1, 0, 0, 0))
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=1).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = sentence_bleu([reference1, reference2, reference3], hypothesis1, weights=(0.5, 0.5, 0, 0))
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=2).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = sentence_bleu(
        [reference1, reference2, reference3], hypothesis1, weights=(0.33333333, 0.33333333, 0.33333333, 0)
    )
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=3).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = sentence_bleu([reference1, reference2, reference3], hypothesis1, weights=(0.25, 0.25, 0.25, 0.25))
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=4).item()
    assert round(pl_output, 4) == round(nltk_output, 4)


def test_with_sentence_bleu_smooth():
    nltk_output = sentence_bleu(
        [reference1, reference2, reference3], hypothesis1, weights=(1, 0, 0, 0), smoothing_function=smooth_func
    )
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=1, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = sentence_bleu(
        [reference1, reference2, reference3], hypothesis1, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_func
    )
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=2, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = sentence_bleu(
        [reference1, reference2, reference3],
        hypothesis1,
        weights=(0.33333333, 0.33333333, 0.33333333, 0),
        smoothing_function=smooth_func,
    )
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=3, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = sentence_bleu(
        [reference1, reference2, reference3],
        hypothesis1,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth_func,
    )
    pl_output = bleu_score([hypothesis1], [[reference1, reference2, reference3]], n=4, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)


# example taken from https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu
hyp1 = [
    "It",
    "is",
    "a",
    "guide",
    "to",
    "action",
    "which",
    "ensures",
    "that",
    "the",
    "military",
    "always",
    "obeys",
    "the",
    "commands",
    "of",
    "the",
    "party",
]
ref1a = [
    "It",
    "is",
    "a",
    "guide",
    "to",
    "action",
    "that",
    "ensures",
    "that",
    "the",
    "military",
    "will",
    "forever",
    "heed",
    "Party",
    "commands",
]
ref1b = [
    "It",
    "is",
    "the",
    "guiding",
    "principle",
    "which",
    "guarantees",
    "the",
    "military",
    "forces",
    "always",
    "being",
    "under",
    "the",
    "command",
    "of",
    "the",
    "Party",
]
ref1c = [
    "It",
    "is",
    "the",
    "practical",
    "guide",
    "for",
    "the",
    "army",
    "always",
    "to",
    "heed",
    "the",
    "directions",
    "of",
    "the",
    "party",
]

hyp2 = ["he", "read", "the", "book", "because", "he", "was", "interested", "in", "world", "history"]
ref2a = ["he", "was", "interested", "in", "world", "history", "because", "he", "read", "the", "book"]

list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
hypotheses = [hyp1, hyp2]


def test_with_corpus_bleu():
    nltk_output = corpus_bleu(list_of_references, hypotheses, weights=(1, 0, 0, 0))
    pl_output = bleu_score(hypotheses, list_of_references, n=1).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = corpus_bleu(list_of_references, hypotheses, weights=(0.5, 0.5, 0, 0))
    pl_output = bleu_score(hypotheses, list_of_references, n=2).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = corpus_bleu(list_of_references, hypotheses, weights=(0.33333333, 0.33333333, 0.33333333, 0))
    pl_output = bleu_score(hypotheses, list_of_references, n=3).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    pl_output = bleu_score(hypotheses, list_of_references, n=4).item()
    assert round(pl_output, 4) == round(nltk_output, 4)


def test_with_corpus_bleu_smooth():
    nltk_output = corpus_bleu(list_of_references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth_func)
    pl_output = bleu_score(hypotheses, list_of_references, n=1, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = corpus_bleu(list_of_references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_func)
    pl_output = bleu_score(hypotheses, list_of_references, n=2, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = corpus_bleu(
        list_of_references, hypotheses, weights=(0.33333333, 0.33333333, 0.33333333, 0), smoothing_function=smooth_func
    )
    pl_output = bleu_score(hypotheses, list_of_references, n=3, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)

    nltk_output = corpus_bleu(
        list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func
    )
    pl_output = bleu_score(hypotheses, list_of_references, n=4, smooth=True).item()
    assert round(pl_output, 4) == round(nltk_output, 4)


hyp = [[]]
ref = [[[]]]


def test_empty_bleu():
    assert bleu_score(hyp, ref) == 0.0


hyps = [["My", "full", "pytorch"]]
refs = [[["My", "full", "pytorch", "test"], ["Completely", "Different"]]]


def test_no_4_gram():
    assert bleu_score(hyps, refs) == 0.0
