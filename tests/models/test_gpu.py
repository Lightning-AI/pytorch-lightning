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
from collections import namedtuple
from unittest.mock import patch

import torch

import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.imports import Batch, Dataset, Example, Field, LabelField
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


@RunIf(min_gpus=2)
def test_multi_gpu_none_backend(tmpdir):
    """Make sure when using multiple GPUs the user can't use `distributed_backend = None`."""
    tutils.set_random_master_port()
    trainer_options = dict(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        gpus=2,
    )

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_gpus=1)
def test_single_gpu_batch_parse():
    trainer = Trainer(gpus=1)

    # non-transferrable types
    primitive_objects = [None, {}, [], 1.0, "x", [None, 2], {"x": (1, 2), "y": None}]
    for batch in primitive_objects:
        data = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
        assert data == batch

    # batch is just a tensor
    batch = torch.rand(2, 3)
    batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
    assert batch.device.index == 0 and batch.type() == "torch.cuda.FloatTensor"

    # tensor list
    batch = [torch.rand(2, 3), torch.rand(2, 3)]
    batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0].device.index == 0 and batch[0].type() == "torch.cuda.FloatTensor"
    assert batch[1].device.index == 0 and batch[1].type() == "torch.cuda.FloatTensor"

    # tensor list of lists
    batch = [[torch.rand(2, 3), torch.rand(2, 3)]]
    batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0][0].device.index == 0 and batch[0][0].type() == "torch.cuda.FloatTensor"
    assert batch[0][1].device.index == 0 and batch[0][1].type() == "torch.cuda.FloatTensor"

    # tensor dict
    batch = [{"a": torch.rand(2, 3), "b": torch.rand(2, 3)}]
    batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0]["a"].device.index == 0 and batch[0]["a"].type() == "torch.cuda.FloatTensor"
    assert batch[0]["b"].device.index == 0 and batch[0]["b"].type() == "torch.cuda.FloatTensor"

    # tuple of tensor list and list of tensor dict
    batch = ([torch.rand(2, 3) for _ in range(2)], [{"a": torch.rand(2, 3), "b": torch.rand(2, 3)} for _ in range(2)])
    batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0][0].device.index == 0 and batch[0][0].type() == "torch.cuda.FloatTensor"

    assert batch[1][0]["a"].device.index == 0
    assert batch[1][0]["a"].type() == "torch.cuda.FloatTensor"

    assert batch[1][0]["b"].device.index == 0
    assert batch[1][0]["b"].type() == "torch.cuda.FloatTensor"

    # namedtuple of tensor
    BatchType = namedtuple("BatchType", ["a", "b"])
    batch = [BatchType(a=torch.rand(2, 3), b=torch.rand(2, 3)) for _ in range(2)]
    batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0].a.device.index == 0
    assert batch[0].a.type() == "torch.cuda.FloatTensor"

    # non-Tensor that has `.to()` defined
    class CustomBatchType:
        def __init__(self):
            self.a = torch.rand(2, 2)

        def to(self, *args, **kwargs):
            self.a = self.a.to(*args, **kwargs)
            return self

    batch = trainer.accelerator.batch_to_device(CustomBatchType(), torch.device("cuda:0"))
    assert batch.a.type() == "torch.cuda.FloatTensor"

    # torchtext.data.Batch
    samples = [
        {"text": "PyTorch Lightning is awesome!", "label": 0},
        {"text": "Please make it work with torchtext", "label": 1},
    ]

    text_field = Field()
    label_field = LabelField()
    fields = {"text": ("text", text_field), "label": ("label", label_field)}

    examples = [Example.fromdict(sample, fields) for sample in samples]
    dataset = Dataset(examples=examples, fields=fields.values())

    # Batch runs field.process() that numericalizes tokens, but it requires to build dictionary first
    text_field.build_vocab(dataset)
    label_field.build_vocab(dataset)

    batch = Batch(data=examples, dataset=dataset)
    batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))

    assert batch.text.type() == "torch.cuda.LongTensor"
    assert batch.label.type() == "torch.cuda.LongTensor"


@RunIf(min_gpus=1)
def test_non_blocking():
    """Tests that non_blocking=True only gets passed on torch.Tensor.to, but not on other objects."""
    trainer = Trainer()

    batch = torch.zeros(2, 3)
    with patch.object(batch, "to", wraps=batch.to) as mocked:
        batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
        mocked.assert_called_with(torch.device("cuda", 0), non_blocking=True)

    class BatchObject:
        def to(self, *args, **kwargs):
            pass

    batch = BatchObject()
    with patch.object(batch, "to", wraps=batch.to) as mocked:
        batch = trainer.accelerator.batch_to_device(batch, torch.device("cuda:0"))
        mocked.assert_called_with(torch.device("cuda", 0))
