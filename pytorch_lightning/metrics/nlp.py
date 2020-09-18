# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from pytorch_lightning.metrics.functional.nlp import bleu_score
from pytorch_lightning.metrics.metric import Metric


class BLEUScore(Metric):
    """
    Calculate BLEU score of machine translated text with one or more references.

    Example:

        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> metric = BLEUScore()
        >>> metric(translate_corpus, reference_corpus)
        tensor(0.7598)
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
            translate_corpus: An iterable of machine translated corpus
            reference_corpus: An iterable of iterables of reference corpus

        Return:
            torch.Tensor: BLEU Score
        """
        return bleu_score(
            translate_corpus=translate_corpus,
            reference_corpus=reference_corpus,
            n_gram=self.n_gram,
            smooth=self.smooth,
        ).to(self.device, self.dtype)
