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


def embedding_similarity(
    batch: torch.Tensor,
    similarity: str = 'cosine',
    reduction: str = 'none',
    zero_diagonal: bool = True
) -> torch.Tensor:
    """
    Computes representation similarity

    Example:

        >>> embeddings = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.], [4., 5., 6., 7.]])
        >>> embedding_similarity(embeddings)
        tensor([[0.0000, 1.0000, 0.9759],
                [1.0000, 0.0000, 0.9759],
                [0.9759, 0.9759, 0.0000]])

    Args:
        batch: (batch, dim)
        similarity: 'dot' or 'cosine'
        reduction: 'none', 'sum', 'mean' (all along dim -1)
        zero_diagonal: if True, the diagonals are set to zero

    Return:
        A square matrix (batch, batch) with the similarity scores between all elements
        If sum or mean are used, then returns (b, 1) with the reduced value for each row
    """
    if similarity == 'cosine':
        norm = torch.norm(batch, p=2, dim=1)
        batch = batch / norm.unsqueeze(1)

    sqr_mtx = batch.mm(batch.transpose(1, 0))

    if zero_diagonal:
        sqr_mtx = sqr_mtx.fill_diagonal_(0)

    if reduction == 'mean':
        sqr_mtx = sqr_mtx.mean(dim=-1)

    if reduction == 'sum':
        sqr_mtx = sqr_mtx.sum(dim=-1)

    return sqr_mtx
