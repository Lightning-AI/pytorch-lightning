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
from torch import tensor

from pytorch_lightning import seed_everything
from pytorch_lightning.core.step_result import DefaultMetricsKeys, ResultCollection


def test_result_collection_on_tensor_with_mean_reduction():

    seed_everything(42)

    result_collection = ResultCollection()

    for i in range(1, 10):
        for prob_bar in [False, True]:
            for logger in [False, True]:
                result_collection.log(
                    "training_step",
                    f"loss_1_{int(prob_bar)}_{int(logger)}",
                    torch.tensor(i),
                    on_step=True,
                    on_epoch=True,
                    batch_size=i**2,
                    prog_bar=prob_bar,
                    logger=logger
                )
                result_collection.log(
                    "training_step",
                    f"loss_2_{int(prob_bar)}_{int(logger)}",
                    torch.tensor(i),
                    on_step=False,
                    on_epoch=True,
                    batch_size=i**2,
                    prog_bar=prob_bar,
                    logger=logger
                )
                result_collection.log(
                    "training_step",
                    f"loss_3_{int(prob_bar)}_{int(logger)}",
                    torch.tensor(i),
                    on_step=True,
                    on_epoch=False,
                    batch_size=i**2,
                    prog_bar=prob_bar,
                    logger=logger
                )
                result_collection.log(
                    "training_step",
                    f"loss_4_{int(prob_bar)}_{int(logger)}",
                    torch.tensor(i),
                    on_step=False,
                    on_epoch=False,
                    batch_size=i**2,
                    prog_bar=prob_bar,
                    logger=logger
                )

    excepted_values = [
        tensor(1), tensor(2),
        tensor(3), tensor(4),
        tensor(5), tensor(6),
        tensor(7), tensor(8),
        tensor(9)
    ]
    excepted_batches = [1, 4, 9, 16, 25, 36, 49, 64, 81]
    total_value = tensor(excepted_values) * tensor(excepted_batches)
    assert result_collection["training_step.loss_1_0_0"].value == sum(total_value)
    assert result_collection["training_step.loss_1_0_0"].cumulated_batch_size == sum(excepted_batches)

    batch_metrics = result_collection.get_batch_metrics()

    expected = {
        'loss_1_1_0_step': tensor([9.]),
        'loss_3_1_0': tensor([9.]),
        'loss_1_1_1_step': tensor([9.]),
        'loss_3_1_1': tensor([9.])
    }
    assert batch_metrics[DefaultMetricsKeys.PBAR] == expected

    excepted = {
        'loss_1_0_1_step': tensor([9.]),
        'loss_3_0_1': tensor([9.]),
        'loss_1_1_1_step': tensor([9.]),
        'loss_3_1_1': tensor([9.])
    }
    assert batch_metrics[DefaultMetricsKeys.LOG] == excepted

    excepted = {
        'loss_1_0_0': tensor([9.]),
        'loss_3_0_0': tensor([9.]),
        'loss_1_0_1': tensor([9.]),
        'loss_3_0_1': tensor([9.]),
        'loss_1_1_0': tensor([9.]),
        'loss_3_1_0': tensor([9.]),
        'loss_1_1_1': tensor([9.]),
        'loss_3_1_1': tensor([9.])
    }
    assert batch_metrics[DefaultMetricsKeys.CALLBACK] == excepted

    epoch_metrics = result_collection.get_epoch_metrics()

    mean = (tensor(excepted_values) * tensor(excepted_batches)).sum() / sum(excepted_batches)

    expected = {'loss_1_1_0_epoch': mean, 'loss_2_1_0': mean, 'loss_1_1_1_epoch': mean, 'loss_2_1_1': mean}
    assert epoch_metrics[DefaultMetricsKeys.PBAR] == expected

    excepted = {'loss_1_0_1_epoch': mean, 'loss_2_0_1': mean, 'loss_1_1_1_epoch': mean, 'loss_2_1_1': mean}
    assert epoch_metrics[DefaultMetricsKeys.LOG] == excepted

    excepted = {
        'loss_1_0_0': mean,
        'loss_2_0_0': mean,
        'loss_1_0_1': mean,
        'loss_2_0_1': mean,
        'loss_1_1_0': mean,
        'loss_2_1_0': mean,
        'loss_1_1_1': mean,
        'loss_2_1_1': mean
    }
    assert epoch_metrics[DefaultMetricsKeys.CALLBACK] == excepted


def test_result_collection_restoration():

    result_collection = ResultCollection(True)

    result_collection