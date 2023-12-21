import torch
from lightning.data.streaming import CombinedStreamingDataset, StreamingDataLoader
from torch import tensor

from tests.tests_data.streaming.test_combined import TestStatefulDataset


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

    assert dataloader.state_dict() == {"0": {"counter": 11}, "1": {"counter": 9}}
