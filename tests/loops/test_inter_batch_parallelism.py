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
import time
from statistics import mean
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tests.helpers.runif import RunIf


def get_cycles_per_ms() -> float:
    """
    Measure and return approximate number of cycles per millisecond for torch.cuda._sleep

    Copied from: github.com/pytorch/pytorch/blob/master/test/test_cuda.py
    """

    def measure() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # Get 10 values and remove the 2 max and 2 min and return the avg.
    # This is to avoid system disturbance that skew the results, e.g.
    # the very first cuda call likely does a bunch of init, which takes
    # much longer than subsequent calls.
    #
    # Tested on both Tesla V100, Quadro GP100, Titan RTX, RTX 3090 GPUs
    # and seems to return stable values. Therefore, we enable caching
    # using lru_cache decorator above.
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2 : num - 2])


_CYCLES_PER_MS = int(get_cycles_per_ms()) if torch.cuda.is_available() else 0
_BATCH_SIZE = 128
_EMB_SZ = 100
_EMB_DIM = 64


class RandomSparseDataset(IterableDataset):
    def __init__(self, emb_dim: int, batch_size: int, count: int) -> None:
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.count = count

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randint(self.emb_dim, [self.batch_size])


class ToyDLRMModel(LightningModule):
    """
    A toy model for mimicking the communication overhead of sharded embedding
    modules in DLRM models.

    DLRM models can be trained in a DDP-like fashion, where each trainer
    receives different batches (embedding indices in this example). Since the
    embeddings are sharded across trainers, the lookup process involves (1)
    routing the indices to the trainer that possesses the corresponding
    embeddings (2) performing local lookup (3) routing the embedding lookup
    result back.

    The toy model doesn't actually performs index/result routing. It simply
    uses torch.cuda._sleep() to mimic the cost of the communication op (i.e.
    a2a).
    """

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.local_embedding = torch.nn.Embedding(_EMB_SZ, _EMB_DIM)

    def _route_indices(self, batch: torch.Tensor, non_blocking=False):
        """
        This can be parallelized across different batches since it's model
        weight independent.

        Why not run this in dataloader/datamodule?
        - The routing logic depends on how model is sharded
        - Putting this in data preprocessor changes the semantic of the model
        """
        torch.cuda._sleep(_CYCLES_PER_MS * 1_000)
        if not non_blocking:
            torch.cuda.synchronize()
        return batch

    def _route_result(self, result: torch.Tensor, non_blocking=False):
        torch.cuda._sleep(_CYCLES_PER_MS * 1_000)
        if not non_blocking:
            torch.cuda.synchronize()
        return result

    def forward(self, indices: torch.Tensor):
        local_indices = self._route_indices(indices)
        result = self.local_embedding(local_indices)
        return self._route_result(result)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        return self.forward(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.local_embedding.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(RandomSparseDataset(_EMB_DIM, _BATCH_SIZE, 5))


class AsyncToyDLRMModel(ToyDLRMModel):
    def __init__(self):
        super().__init__()
        self.comm_stream = torch.cuda.Stream()
        self.batch_i = None
        self.batch_i_ready = torch.cuda.Event()

    def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:
        if self.batch_i is None:
            self.batch_i = next(dataloader_iter)
            with torch.cuda.stream(self.comm_stream):
                self._route_indices(self.batch_i, non_blocking=True)
                self.batch_i_ready.record()

        # Invariant: the routing for batch[i] has been kicked off
        is_last = False
        batch_ip1 = None
        batch_ip1_ready = torch.cuda.Event()
        try:
            batch_ip1 = next(dataloader_iter)
            with torch.cuda.stream(self.comm_stream):
                self._route_indices(batch_ip1, non_blocking=True)
                batch_ip1_ready.record()
        except StopIteration:
            is_last = True

        self.batch_i_ready.wait()

        result = self.local_embedding(self.batch_i)
        self._route_result(result)

        self.batch_i = batch_ip1
        self.batch_i_ready = batch_ip1_ready

        return {"is_last": is_last}


@RunIf(min_gpus=1)
def test_inter_batch_parallelism(tmpdir):
    """
    Verify the speedup of a simple inter-batch parallelization use case enabled
    by exposing `dataloader_iter` to `training_step`.
    """
    begin_time = time.time()
    m = AsyncToyDLRMModel()
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(m)
    async_duration = time.time() - begin_time

    begin_time = time.time()
    m = ToyDLRMModel()
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(m)
    sync_duration = time.time() - begin_time

    # We expect 2x speedup. However, we only assert that the async
    # training_step is faster in order to avoid flaky tests
    assert async_duration < sync_duration, "Expect `AsyncToyDLRMModel` to train faster than `ToyDLRMModel`."
