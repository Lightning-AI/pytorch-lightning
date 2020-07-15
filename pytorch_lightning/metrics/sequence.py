from typing import List

from pytorch_lightning.metrics.functional.sequence import bleu_score
from pytorch_lightning.metrics.metric import Metric


class BLEUScore(Metric):
    """
    Calculate BLEU score of machine translated text with one or more references.

    Example:

        >>> translate_corpus = ["the FAST brown fox jumped over the lazy dog".split(' ')]
        >>> reference_corpus = [["the quick brown fox jumped over the lazy dog".split(' '), "the quick brown fox jumped over the the lazy cat".split(' ')]]
        >>> metric = BLEUScore()
        >>> metric(translate_corpus, reference_corpus)
        0.750623881816864
    """

    def __init__(self, n_gram: int = 4, weights: List[float] = [0.25] * 4):
        """
        Args:
            n_gram: Gram value ranged from 1 to 4
            weights: A list of weights used for each n-gram category (uniform by default)
        """
        super().__init__(name="bleu")
        self.n_gram = n_gram
        self.weights = weights

    def forward(self, translate_corpus: List[str], reference_corpus: List[str]) -> float:
        """
        Actual metric computation

        Args:
            translate_corpus: A list of lists of machine translated corpus
            reference_corpus: A list of lists of reference corpus

        Return:
            float: BLEU Score
        """
        return bleu_score(
            translate_corpus=translate_corpus,
            reference_corpus=reference_corpus,
            n_gram=self.n_gram,
            weights=self.weights,
        )
