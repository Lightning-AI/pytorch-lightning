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


def retrieval_hit_rate(
    preds: torch.Tensor,
    target: torch.Tensor,
    k: int = 1
) -> torch.Tensor:
    """
    Computes the hit_rate @ k metric for information retrieval
    Hir Rate at k is 1 iff there is at least one relevant documents among the top K.

    `preds` and `target` should be of the same shape and live on the same device. If not target is true, 0 is returned.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant.
        k: consider only the top k elements.

    Returns:
        a single-value tensor with the hit rate at k (HR@K) of the predictions `preds` wrt the labels `target`.

    Example:

        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([True, False, True])
        >>> retrieval_hit_rate(preds, target, k=2)
        ... 1.0
    """

    if preds.shape != target.shape or preds.device != target.device: 
        raise ValueError(
            f"`preds` and `target` must have the same shape and be on the same device"
        )

    relevant = target[torch.argsort(preds, dim=-1, descending=True)][:k].sum()
    return (relevant > 0).to(preds)
