import pytest
from lightning.data.streaming.combined import CombinedStreamingDataset
from torch.utils.data import IterableDataset


def test_combined_dataset_num_samples_yield():
    dataset = CombinedStreamingDataset([range(10), range(0, -10, -1)], 42, weights=(0.5, 0.5))
    dataset_iter = iter(dataset)

    data = list(dataset_iter)
    assert data == [0, 0, 1, 2, -1, -2, -3, 3, 4, 5, 6, -4, 7, 8, -5, -6, 9, -7, -8]

    dataset = CombinedStreamingDataset([range(10), range(0, -10, -1)], 37, weights=(0.5, 0.5))
    dataset_iter = iter(dataset)

    data = list(dataset_iter)
    assert data == [0, 0, -1, -2, -3, -4, -5, 1, -6, 2, -7, -8, 3, 4, -9, 5]

    dataset = CombinedStreamingDataset([range(10), range(0, -10, -1)], 23, weights=(0.5, 0.5))
    dataset_iter = iter(dataset)

    data = [next(dataset_iter) for _ in range(5)]
    assert data == [0, -1, -2, 0, -3]
    assert dataset._iterator._num_samples_yielded == [1, 4]
    assert next(dataset_iter) == 1
    assert dataset._iterator._num_samples_yielded == [2, 4]


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


def test_combined_dataset_state_dict():
    dataset = CombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5)
    )
    assert dataset.state_dict(0, 1) == {}
    dataset_iter = iter(dataset)
    assert dataset.state_dict(0, 1) == {"0": {"counter": 0}, "1": {"counter": 0}}

    dataset2 = CombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5)
    )
    assert dataset2.state_dict(0, 1) == {}

    data = []
    states = []
    for i, value in enumerate(dataset_iter):
        state = dataset.state_dict(i, 1)
        data.append(value)
        states.append(state)

    assert data == [0, 0, 1, 2, -1, -2, -3, 3, 4, 5, 6, -4, 7, 8, -5, -6, 9, -7, -8]
    assert states == [
        {"0": {"counter": 0}, "1": {"counter": 1}},
        {"0": {"counter": 1}, "1": {"counter": 1}},
        {"0": {"counter": 2}, "1": {"counter": 1}},
        {"0": {"counter": 3}, "1": {"counter": 1}},
        {"0": {"counter": 3}, "1": {"counter": 2}},
        {"0": {"counter": 3}, "1": {"counter": 3}},
        {"0": {"counter": 3}, "1": {"counter": 4}},
        {"0": {"counter": 4}, "1": {"counter": 4}},
        {"0": {"counter": 5}, "1": {"counter": 4}},
        {"0": {"counter": 6}, "1": {"counter": 4}},
        {"0": {"counter": 7}, "1": {"counter": 4}},
        {"0": {"counter": 7}, "1": {"counter": 5}},
        {"0": {"counter": 8}, "1": {"counter": 5}},
        {"0": {"counter": 9}, "1": {"counter": 5}},
        {"0": {"counter": 9}, "1": {"counter": 6}},
        {"0": {"counter": 9}, "1": {"counter": 7}},
        {"0": {"counter": 10}, "1": {"counter": 7}},
        {"0": {"counter": 10}, "1": {"counter": 8}},
        {"0": {"counter": 10}, "1": {"counter": 9}},
    ]

    dataset2 = CombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5)
    )
    assert dataset2.state_dict(0, 1) == {}
    dataset2_iter = iter(dataset2)

    data_2 = []
    for state in states:
        dataset.load_state_dict(state)
        data_2.append(next(dataset2_iter))

    assert data == data_2


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([1], [1]),
        ([2], [1]),
        ([2, 0.5], [0.8, 0.2]),
        ([1, 1, 1], [1 / 3, 1 / 3, 1 / 3]),
        ([0.3, 0, 0], [1.0, 0, 0]),
        (None, [0.5, 0.5]),
    ],
)
def test_combined_dataset_normalizes_weights(weights, expected):
    combined_dataset = CombinedStreamingDataset([[1], [2, 3]], weights=weights, seed=1)
    assert combined_dataset._weights == expected


class SimpleDataset(IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        self._start = start
        self._end = end

    def __iter__(self):
        return iter(range(self._start, self._end))


def test_combined_dataset():
    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[1.0, 0.0], seed=12345)

    res = list(dataset)
    assert res == list(range(0, 10))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[0.0, 1.0], seed=12345)

    res = list(dataset)
    assert res == list(range(10, 20))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[0.5, 0.5], seed=12345)

    res = list(dataset)
    assert 9 in res or 19 in res
    if len(res) > 10:
        assert 0 in res
        assert 10 in res
