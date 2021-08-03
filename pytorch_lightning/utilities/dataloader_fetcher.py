from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Any
import pytorch_lightning as pl
import torch


@dataclass()
class LightningStreamEvent:

    inter_batch_parallelism: bool
    device: torch.device

    def __post_init__(self):
        if self.cuda_inter_batch_parallelism:
            self.cuda_stream = torch.cuda.Stream()

    @contextmanager
    def stream_context(self):
        self.start_record()
        if self.cuda_inter_batch_parallelism:
            with torch.cuda.stream(self.cuda_stream):
                yield
        else:
            yield
        self.end_record()

    @property
    def cuda_inter_batch_parallelism(self) -> bool:
        return self.device.type == "cuda" and self.inter_batch_parallelism

    def start_record(self) -> None:
        if self.cuda_inter_batch_parallelism:
            self.event = torch.cuda.Event()

    def end_record(self) -> None:
        if self.cuda_inter_batch_parallelism:
            self.event.record()

    def wait(self) -> None:
        if self.cuda_inter_batch_parallelism:
            self.event.wait()


def profiled_iterator(iterator, profiler):
    with profiler.profile("get_train_batch"):
        yield next(iterator)


class LightningFetcher(object):

    def __init__(self, datalaoder, inter_batch_parallelism: bool, batch_to_device: Callable, profiler: 'pl.profiler.base.BaseProfiler', device: torch.device) -> None:
        self.datalaoder = datalaoder
        self.stream = LightningStreamEvent(inter_batch_parallelism, device)
        self.profiler = profiler
        self.batch_to_device = batch_to_device

    def __iter__(self) -> "LightningFetcher":
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
