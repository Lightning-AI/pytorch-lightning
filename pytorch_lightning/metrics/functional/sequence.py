from collections import Counter
from typing import List

import torch
from torchtext.data.metrics import bleu_score as bleu


def bleu_score(
    translate_corpus: List[str], reference_corpus: List[str], n_gram: int = 4, weights: List[float] = [0.25] * 4,
) -> float:
    """Calculate BLEU score of machine translated text with one or more references.

    Args:
        translate_corpus: A list of lists of machine translated corpus
        reference_corpus: A list of lists of reference corpus
        n_gram: Gram value ranged from 1 to 4
        weights: A list of weights used for each n-gram category (uniform by default)

    Return:
        A BLEU Score

    Example:

        >>> translate_corpus = ["the FAST brown fox jumped over the lazy dog".split(' ')]
        >>> reference_corpus = [["the quick brown fox jumped over the lazy dog".split(' '), "the quick brown fox jumped over the the lazy cat".split(' ')]]
        >>> bleu_score(translate_corpus, reference_corpus)
        0.750623881816864
    """
    return bleu(candidate_corpus=translate_corpus, references_corpus=reference_corpus, max_n=n_gram, weights=weights)
