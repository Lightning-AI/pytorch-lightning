import torch
from lightning.data.streaming import CombinedStreamingDataset, StreamingDataLoader
from torch import tensor


class TestStatefulDataset:
    def __init__(self, size, step):
        self.size = size
        self.step = step
        self.counter = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == self.size:
            raise StopIteration
        value = self.step * self.counter
        self.counter += 1
        return value

    def state_dict(self, *args, **kwargs):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]


def test_streaming_dataloader():
    dataset = CombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5)
    )
    dataloader = StreamingDataLoader(dataset, batch_size=2)
    dataloader_iter = iter(dataloader)
    batches = []
    for batch in dataloader_iter:
        batches.append(batch)

    expected = [
        tensor([0, 0]),
        tensor([1, 2]),
        tensor([-1, -2]),
        tensor([-3, 3]),
        tensor([4, 5]),
        tensor([6, -4]),
        tensor([7, 8]),
        tensor([-5, -6]),
        tensor([9, -7]),
        tensor([-8]),
    ]

    for exp, gen in zip(expected, batches):
        assert torch.equal(exp, gen)

    assert dataloader.state_dict() == {"0": {"counter": 10}, "1": {"counter": 9}}
