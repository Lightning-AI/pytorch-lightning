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
"""Test deprecated functionality which will be removed in v1.5.0"""

import pytest
import torch

from pytorch_lightning.metrics import Accuracy, MetricCollection
from pytorch_lightning.metrics.utils import get_num_classes, select_topk, to_categorical, to_onehot


def test_v1_5_metrics_utils():
    x = torch.tensor([1, 2, 3])
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(to_onehot(x), torch.Tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(int))

    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert get_num_classes(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 0])) == 4

    x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(select_topk(x, topk=2), torch.Tensor([[0, 1, 1], [1, 1, 0]]).to(torch.int32))

    x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(to_categorical(x), torch.Tensor([1, 0]).to(int))


def test_v1_5_metrics_collection():
    target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
    preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])
    with pytest.deprecated_call(
        match="`pytorch_lightning.metrics.metric.MetricCollection` was deprecated since v1.3.0 in favor"
              " of `torchmetrics.collections.MetricCollection`. It will be removed in v1.5.0."
    ):
        metrics = MetricCollection([Accuracy()])
    assert metrics(preds, target) == {'Accuracy': torch.tensor(0.1250)}


def test_v1_5_metric_accuracy():
    from pytorch_lightning.metrics.functional.accuracy import accuracy
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert accuracy(preds=torch.tensor([0, 1]), target=torch.tensor([0, 1])) == torch.tensor(1.)

    from pytorch_lightning.metrics import Accuracy
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        Accuracy()
