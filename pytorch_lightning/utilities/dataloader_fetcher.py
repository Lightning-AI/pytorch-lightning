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
from typing import Any, Callable, Generator, List

import torch

import pytorch_lightning as pl


def profiled_iterator(iterable, profiler):
    iterator = iter(iterable)
    while True:
        try:
            with profiler.profile("get_train_batch"):
                yield next(iterator)
        except StopIteration:
            return


class LightningFetcher:

    """
    This class is used to perform ``pre-fetching`` for the ``train`` dataloader
    and apply inter batch parallelism if enabled.

    batch 0: [HtoD][forward][backward]
    batch 1:                          [HtoD][forward][backward]
    With parallelization, the latency of HtoD copy can be hidden:

    batch 0: [HtoD][forward][backward]
    batch 1:       [HtoD]             [forward][backward]
    """

    def __init__(
        self,
        dataloader,
        batch_to_device: Callable,
        profiler: "pl.profiler.base.BaseProfiler",
        num_prefetch_batches: int = 1,
    ) -> None:
        self.iterator = profiled_iterator(dataloader, profiler)
        self.batch_to_device = batch_to_device
        self.batches: List = []
        self.events: List = []
        self.counter: int = 0
        self.num_prefetch_batches = num_prefetch_batches

    def __iter__(self) -> Generator:
        self.counter = 1
        return self.prefetch_function()

    def add_event(self, event) -> None:
        self.events.append(event)

    def add_batch(self, batch) -> None:
        self.batches.append(batch)

    @staticmethod
    def start_record(event) -> None:
        event.record()

    def fetch_batch(self):
        return self.batches.pop(0)

    def wait(self) -> None:
        event = self.events.pop(0)
        event.wait()

    def prefetch_function(self) -> Any:
        cuda_stream = torch.cuda.Stream()

        done = False
        while not done:

            for _ in range(self.num_prefetch_batches + 1):
                if not done:
                    with torch.cuda.stream(cuda_stream):
                        batch_event = torch.cuda.Event()
                        self.add_event(batch_event)
                        try:
                            batch = next(self.iterator)
                            batch = self.batch_to_device(batch)
                            self.add_batch(batch)
                            self.start_record(batch_event)
                        except StopIteration:
                            done = True

            self.wait()
            batch = self.fetch_batch()
            # yield last and has next
            yield self.counter, batch, done
            self.counter += 1
