from lightning.data.streaming.combined import CombinedStreamingDataset


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


def test_combined_dataset_state_dict():
    class TestDataset:
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

        def state_dict(self, sample):
            return {"sample": sample}

    dataset = CombinedStreamingDataset([TestDataset(10, 1), TestDataset(10, -1)], 42, weights=(0.5, 0.5))
    assert dataset.state_dict() == {}
    dataset_iter = iter(dataset)
    assert dataset.state_dict() == {"0": {"sample": 0}, "1": {"sample": 0}}

    data = []
    states = []
    for value in dataset_iter:
        data.append(value)
        states.append(dataset.state_dict())

    assert data == [0, 0, 1, 2, -1, -2, -3, 3, 4, 5, 6, -4, 7, 8, -5, -6, 9, -7, -8]
    assert states == [
        {"0": {"sample": 0}, "1": {"sample": 1}},
        {"0": {"sample": 1}, "1": {"sample": 1}},
        {"0": {"sample": 2}, "1": {"sample": 1}},
        {"0": {"sample": 3}, "1": {"sample": 1}},
        {"0": {"sample": 3}, "1": {"sample": 2}},
        {"0": {"sample": 3}, "1": {"sample": 3}},
        {"0": {"sample": 3}, "1": {"sample": 4}},
        {"0": {"sample": 4}, "1": {"sample": 4}},
        {"0": {"sample": 5}, "1": {"sample": 4}},
        {"0": {"sample": 6}, "1": {"sample": 4}},
        {"0": {"sample": 7}, "1": {"sample": 4}},
        {"0": {"sample": 7}, "1": {"sample": 5}},
        {"0": {"sample": 8}, "1": {"sample": 5}},
        {"0": {"sample": 9}, "1": {"sample": 5}},
        {"0": {"sample": 9}, "1": {"sample": 6}},
        {"0": {"sample": 9}, "1": {"sample": 7}},
        {"0": {"sample": 10}, "1": {"sample": 7}},
        {"0": {"sample": 10}, "1": {"sample": 8}},
        {"0": {"sample": 10}, "1": {"sample": 9}},
    ]
