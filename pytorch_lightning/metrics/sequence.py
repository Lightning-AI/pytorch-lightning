import torch

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
        tensor(0.7506)
    """

    def __init__(self, n_gram: int = 4, smooth: bool = False):
        """
        Args:
            n_gram: Gram value ranged from 1 to 4 (Default 4)
            smooth: Whether or not to apply smoothing – Lin et al. 2004
        """
        super().__init__(name="bleu")
        self.n_gram = n_gram
        self.smooth = smooth

    def forward(self, translate_corpus: list, reference_corpus: list) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            translate_corpus: machine translated corpus
            reference_corpus: reference corpus

        Return:
            torch.Tensor: BLEU Score
        """
        return bleu_score(
            translate_corpus=translate_corpus,
            reference_corpus=reference_corpus,
            n_gram=self.n_gram,
            smooth=self.smooth,
        )
