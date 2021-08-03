from contextlib import contextmanager
import time
from dataclasses import dataclass
from typing import Callable, Any, Generator, List
import pytorch_lightning as pl
import torch

import pytorch_lightning as pl


@dataclass()
class LightningStreamEvent:

    """
    This class is used to capture the device stream and report its associated event
    """

    inter_batch_parallelism: bool
    device: torch.device

    def __post_init__(self):
        if self.cuda_inter_batch_parallelism:
            self.cuda_stream = torch.cuda.Stream()
            self.events: List[torch.cuda.Event] = []

    @contextmanager
    def stream_context(self):
        if self.cuda_inter_batch_parallelism:
            with torch.cuda.stream(self.cuda_stream):
                self.start_record()
                yield
                self.end_record()
        else:
            self.start_record()
            yield
            self.end_record()

    @property
    def cuda_inter_batch_parallelism(self) -> bool:
        return self.device.type == "cuda" and self.inter_batch_parallelism

    def start_record(self) -> None:
        if self.cuda_inter_batch_parallelism:
            self.events.append(torch.cuda.Event())

    def end_record(self) -> None:
        if self.cuda_inter_batch_parallelism:
            self.events[-1].record()

    def wait(self) -> None:
        if self.cuda_inter_batch_parallelism:
            event = self.events.pop(0)
            t0 = time.time()
            event.wait()
            time()


def profiled_iterator(iterator, profiler):
    with profiler.profile("get_train_batch"):
        yield next(iterator)


class LightningFetcher(object):

    """
    This class is used to perform ``pre-fecthing`` for the ``train`` dataloader and apply iter batch parallelism if enabled. 

    batch 0: [HtoD][forward][backward]
    batch 1:                          [HtoD][forward][backward]
    With parallelization, the latency of HtoD copy can be hidden:

    batch 0: [HtoD][forward][backward]
    batch 1:       [HtoD]             [forward][backward]
    """

    def __init__(
        self,
        datalaoder,
        inter_batch_parallelism: bool,
        batch_to_device: Callable,
        profiler: 'pl.profiler.base.BaseProfiler',
        device: torch.device,
        num_prefetch_batch: int = 1,
    ) -> None:
        self.datalaoder = datalaoder
        self.stream = LightningStreamEvent(inter_batch_parallelism, device)
        self.profiler = profiler
        self.batch_to_device = batch_to_device
        
        self.num_prefetch_batch = num_prefetch_batch
        if num_prefetch_batch != 1:
            raise NotImplementedError

    def __iter__(self) -> Generator:
        self.iterator = profiled_iterator(iter(self.datalaoder), self.profiler)
        self.counter = 1
        return self.prefect_function()

    def apply_stream(self, batch) -> Any:
        with self.stream.stream_context():
            return self.batch_to_device(batch)

    def prefect_function(self) -> Any:
        try:
            last = self.apply_stream(next(self.iterator))
        except StopIteration:
            return

        for val in self.iterator:
            val = self.apply_stream(val)

            # yield last and has next
            yield self.counter, last, False, self.stream

            # prepare for next batch
            last = val
            self.counter += 1

        # yield last, no longer has next
        yield self.counter, self.apply_stream(last), True, self.stream

    @property
    def wait(self) -> None:
        self.stream.wait()
