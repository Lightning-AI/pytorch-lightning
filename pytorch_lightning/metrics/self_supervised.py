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

from typing import Any

import torch

from pytorch_lightning.metrics.functional.self_supervised import embedding_similarity
from pytorch_lightning.metrics.metric import TensorMetric


__all__ = ['EmbeddingSimilarity']


class EmbeddingSimilarity(TensorMetric):
    """
    Computes similarity between embeddings

    Example:
        >>> embeddings = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.], [4., 5., 6., 7.]])
        >>> embedding_similarity(embeddings)
        tensor([[0.0000, 1.0000, 0.9759],
                [1.0000, 0.0000, 0.9759],
                [0.9759, 0.9759, 0.0000]])

    """
    def __init__(
            self,
            similarity: str = 'cosine',
            zero_diagonal: bool = True,
            reduction: str = 'mean',
            reduce_group: Any = None
    ):
        """
        Args:
            similarity: 'dot' or 'cosine'
            reduction: 'none', 'sum', 'mean' (all along dim -1)
            zero_diagonal: if True, the diagonals are set to zero
            reduce_group: the process group to reduce metric results from DDP

        """
        super().__init__(name='embedding_similarity',
                         reduce_group=reduce_group)
        self.similarity = similarity
        self.zero_diagonal = zero_diagonal
        self.reduction = self.reduction

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            batch: tensor containing embeddings with shape (batch_size, dim)

        Return:
            A square matrix (batch, batch) with the similarity scores between all elements
            If sum or mean are used, then returns (b, 1) with the reduced value for each row
        """
        return embedding_similarity(batch,
                                    similarity=self.similarity,
                                    zero_diagonal=self.zero_diagonal,
                                    reduction=self.reduction)
