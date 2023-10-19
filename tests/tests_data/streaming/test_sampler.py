from unittest import mock

import pytest
from lightning import seed_everything
from lightning.data.streaming.sampler import CacheBatchSampler


@pytest.mark.parametrize(
    "params",
    [
        (
            21,
            1,
            [[0, 1, 2], [7, 8, 9], [14, 15, 16], [3, 4, 5], [10, 11, 12], [17, 18, 19], [6], [13], [20]],
            [[7, 0, 0], [1, 1, 1], [5, 5, 5], [0, 4, 4], [8, 3, 3], [2, 2, 2], [4], [3], [6]],
        ),
        (
            11,
            1,
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [], [], [9, 10]],
            [[1, 1, 1], [3, 3], [0, 0, 0], [2, 2, 2]],
        ),
        (8, 1, [[0, 1], [2, 3], [4, 5, 6], [], [], [7]], [[1, 1, 2], [3], [0, 0], [2, 2]]),
        (4, 1, [[0], [1], [2, 3]], [[0], [1], [2, 2]]),
        (
            9,
            1,
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        ),
        (
            19,
            1,
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [3, 4, 5], [9, 10, 11], [15, 16, 17], [], [], [18]],
            [[0, 0, 0], [1, 1, 1], [5, 5, 5], [2, 2, 2], [4, 4, 4], [3, 3, 3], [6]],
        ),
        (19, 2, [[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 0, 0], [5, 5, 5], [4, 4, 4], [6]]),
    ],
)
def test_cache_batch_sampler(params):
    seed_everything(42)

    cache = mock.MagicMock()
    cache.filled = False
    if params[1] > 1:
        batch_sampler = CacheBatchSampler(params[0], params[1], 0, 3, 3, False, True, cache)
        batches = []
        for batch in batch_sampler:
            batches.append(batch)
        assert batches == params[2], batches

        batch_sampler = CacheBatchSampler(params[0], 1, 0, 3, 3, False, True, cache)
        batches = []
        for batch in batch_sampler:
            batches.append(batch)

        chunks_interval = [[batch[0], batch[-1] + 1] for batch in batches if len(batch)]
    else:
        batch_sampler = CacheBatchSampler(params[0], params[1], 0, 3, 3, False, True, cache)
        batches = []
        for batch in batch_sampler:
            batches.append(batch)
        assert batches == params[2], batches

        chunks_interval = [[batch[0], batch[-1] + 1] for batch in batches if len(batch)]

    cache.filled = True
    cache.get_chunk_interval.return_value = chunks_interval

    seed_everything(42)

    batch_sampler = CacheBatchSampler(params[0], params[1], 0, 3, 3, False, True, cache)

    batches_1 = []
    for batch in batch_sampler:
        batches_1.append(batch)

    def validate_batch(data, check_values):
        if params[1] == 1:
            assert all(b[0].chunk_indexes is not None for b in data[:3])
            assert all(b[1].chunk_indexes is None if len(b) > 1 else True for b in data[:3])
            assert all(b[0].chunk_indexes is None if len(b) else True for b in data[3:])
            if check_values:
                assert [[x.chunk_index for x in d] for d in data] == params[3]
        else:
            assert all(b[0].chunk_indexes is not None for b in data[:3])
            assert all(b[1].chunk_indexes is None if len(b) > 1 else True for b in data[:3])
            assert all(b[0].chunk_indexes is None if len(b) else True for b in data[3:])
            if check_values:
                assert [[x.chunk_index for x in d] for d in data] == params[3]

    validate_batch(batches_1, True)

    batches_2 = []
    for batch in batch_sampler:
        batches_2.append(batch)

    validate_batch(batches_2, False)
    if params[1] == 1:
        assert batches_1 != batches_2


def test_batch_sampler_imagenet():
    """Validate the Imagenet dataset is valid."""
    dataset_size = 1281167
    world_size = 1
    rank = 0
    num_workers = 32
    batch_size = 8
    cache = mock.MagicMock()
    cache.filled = False
    CacheBatchSampler(dataset_size, world_size, rank, num_workers, batch_size, False, True, cache)
