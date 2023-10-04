from unittest import mock

import numpy as np
import pytest
from lightning import seed_everything
from lightning.data.cache.sampler import CacheBatchSampler, CacheSampler, DistributedCacheSampler


def test_cache_sampler_sampling():
    """Valides the CacheSampler can return batch of data in an ordered way."""
    dataset_size = 17
    sampler = CacheSampler(dataset_size, 3, 3)
    iter_sampler = iter(sampler)

    all_indexes = []
    indexes = []
    while True:
        try:
            index = next(iter_sampler)
            indexes.append(index)
            all_indexes.append(index)
        except StopIteration:
            assert indexes == [0, 1, 2, 5, 6, 7, 10, 11, 12, 3, 4]
            assert sampler._done == {0}
            break

    indexes = []
    while True:
        try:
            index = next(iter_sampler)
            indexes.append(index)
            all_indexes.append(index)
        except StopIteration:
            assert indexes == [8, 9]
            assert sampler._done == {0, 1}
            break

    indexes = []
    while True:
        try:
            index = next(iter_sampler)
            indexes.append(index)
            all_indexes.append(index)
        except StopIteration:
            assert indexes == [13, 14, 15, 16]
            assert sampler._done == {0, 1, 2}
            break

    assert sorted(all_indexes) == list(range(dataset_size))


@pytest.mark.parametrize(
    "params",
    [
        (21, range(0, 7), range(7, 14), range(14, 21)),
        (23, range(0, 7), range(7, 14), range(14, 23)),
        (33, range(0, 11), range(11, 22), range(22, 33)),
        (49, range(0, 16), range(16, 32), range(32, 49)),
        (5, range(0, 1), range(1, 2), range(2, 5)),
        (12, range(0, 4), range(4, 8), range(8, 12)),
    ],
)
def test_cache_sampler_samplers(params):
    sampler = CacheSampler(params[0], 3, 3)
    assert sampler.samplers[0].data_source == params[1]
    assert sampler.samplers[1].data_source == params[2]
    assert sampler.samplers[2].data_source == params[3]


@pytest.mark.parametrize(
    "params",
    [
        (
            102,
            2,
            [
                [range(0, 17), range(17, 34), range(34, 51)],
                [range(51, 68), range(68, 85), range(85, 102)],
            ],
        ),
        (
            227,
            5,
            [
                [range(0, 15), range(15, 30), range(30, 45)],
                [range(45, 60), range(60, 75), range(75, 90)],
                [range(90, 105), range(105, 120), range(120, 135)],
                [range(135, 150), range(150, 165), range(165, 180)],
                [range(180, 195), range(195, 210), range(210, 227)],
            ],
        ),
        (
            1025,
            7,
            [
                [range(0, 48), range(48, 96), range(96, 146)],
                [range(146, 194), range(194, 242), range(242, 292)],
                [range(292, 340), range(340, 388), range(388, 438)],
                [range(438, 486), range(486, 534), range(534, 584)],
                [range(584, 632), range(632, 680), range(680, 730)],
                [range(730, 778), range(778, 826), range(826, 876)],
                [range(876, 924), range(924, 972), range(972, 1025)],
            ],
        ),
        (
            323,
            2,
            [
                [range(0, 53), range(53, 106), range(106, 161)],
                [range(161, 214), range(214, 267), range(267, 323)],
            ],
        ),
        (
            23,
            3,
            [
                [range(0, 2), range(2, 4), range(4, 7)],
                [range(7, 9), range(9, 11), range(11, 14)],
                [range(14, 16), range(16, 18), range(18, 23)],
            ],
        ),
        (
            45,
            2,
            [
                [range(0, 7), range(7, 14), range(14, 22)],
                [range(22, 29), range(29, 36), range(36, 45)],
            ],
        ),
    ],
)
def test_cache_distributed_sampler_samplers(params):
    """This test validates the sub-samplers of the DistributedCacheSampler has the right sampling intervals."""
    for rank in range(params[1]):
        sampler = DistributedCacheSampler(params[0], params[1], rank, 3, 3)
        assert sampler.samplers[0].data_source == params[2][rank][0]
        assert sampler.samplers[1].data_source == params[2][rank][1]
        assert sampler.samplers[2].data_source == params[2][rank][2]


@pytest.mark.parametrize(
    "params",
    [
        (21, 1, [[0, 1, 2], [7, 8, 9], [14, 15, 16], [3, 4, 5], [10, 11, 12], [17, 18, 19], [6], [13], [20]]),
        (11, 1, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [], [], [9, 10]]),
        (8, 1, [[0, 1], [2, 3], [4, 5, 6], [7]]),
        (4, 1, [[0], [1], [2, 3]]),
        (9, 1, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [], [], []]),
        (19, 1, [[0, 1, 2], [6, 7, 8], [12, 13, 14], [3, 4, 5], [9, 10, 11], [15, 16, 17], [], [], [18]]),
        (19, 2, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [], [], []]),
    ],
)
def test_cache_batch_sampler(params):
    cache = mock.MagicMock()
    cache.filled = False
    batch_sampler = CacheBatchSampler(params[0], params[1], 0, 3, 3, False, True, cache)
    batches = []
    for batch in batch_sampler:
        batches.append(batch)
    assert batches == params[2]

    chunk_interval = [[batch[0], batch[-1] + 1] for batch in batches if len(batch)]

    cache.filled = True
    cache.get_chunk_interval.return_value = chunk_interval

    seed_everything(42)

    batch_sampler = CacheBatchSampler(params[0], params[1], 0, 3, 3, False, True, cache)

    batches_1 = []
    for batch in batch_sampler:
        batches_1.extend(batch)

    def validate_batch(data):
        chunks = batch_sampler._shuffled_chunk_intervals
        if params[1] == 1:
            size = 0
            for interval in chunks:
                interval_indices = np.arange(interval[0], interval[1])
                for indice in interval_indices:
                    assert indice in [b.index for b in data[size : size + len(interval_indices)]]
                size += len(interval_indices)
        else:
            chunks_per_replica = len(chunks) // params[1]
            for replica_idx in range(params[1]):
                if replica_idx != 0:
                    continue
                is_last_replica = replica_idx == params[1] - 1
                start_replica = replica_idx * chunks_per_replica
                end_replica = len(chunks) if is_last_replica else (replica_idx + 1) * chunks_per_replica
                shuffled_chunk_intervals_replica = chunks[start_replica:end_replica]

                assert len(shuffled_chunk_intervals_replica)

                size = 0
                for interval in shuffled_chunk_intervals_replica:
                    interval_indices = np.arange(interval[0], interval[1])
                    for indice in interval_indices:
                        assert indice in [b.index for b in data[size : size + len(interval_indices)]]
                    size += len(interval_indices)

    validate_batch(batches_1)
    if params[1] == 1:
        assert len(batches_1) == params[0]

    batches_2 = []
    for batch in batch_sampler:
        batches_2.extend(batch)

    validate_batch(batches_2)
    if params[1] == 1:
        assert batches_1 != batches_2


def test_batch_sampler_imagenet():
    cache = mock.MagicMock()
    cache.filled = False
    CacheBatchSampler(1_281_167, 1, 0, 32, 8, False, False, cache)
