from typing import List, Sequence

try:
    from torchtext.data.metrics import bleu_score as bleu
except ImportError:  # pragma: no-cover
    _TORCHTEXT_AVAILABLE = False
else:
    _TORCHTEXT_AVAILABLE = True


def bleu_score(
    translate_corpus: Sequence, reference_corpus: Sequence, n_gram: int = 4, weights: List[float] = [0.25] * 4,
) -> float:
    """Calculate BLEU score of machine translated text with one or more references.

    Args:
        translate_corpus: An iterable of machine translated corpus
        reference_corpus: An iterable of iterables of reference corpus
        n_gram: Gram value ranged from 1 to 4
        weights: A list of weights used for each n-gram category (uniform by default)

    Return:
        A BLEU score

    Example:

        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> bleu_score(translate_corpus, reference_corpus)
        0.7598356604576111
    """
    if not _TORCHTEXT_AVAILABLE:
        raise ImportError(
            "Using BLEU Score Metric requires `torchtext` to be installed, install it with `pip install torchtext`."
        )
    return bleu(candidate_corpus=translate_corpus, references_corpus=reference_corpus, max_n=n_gram, weights=weights)
