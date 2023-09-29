from unittest import mock

import pytest
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
        (21, [[0, 1, 2], [7, 8, 9], [14, 15, 16], [3, 4, 5], [10, 11, 12], [17, 18, 19], [6], [13], [20]]),
        (11, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [], [], [9, 10]]),
        (8, [[0, 1], [2, 3], [4, 5, 6], [7]]),
        (4, [[0], [1], [2, 3]]),
        (9, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [], [], []]),
        (19, [[0, 1, 2], [6, 7, 8], [12, 13, 14], [3, 4, 5], [9, 10, 11], [15, 16, 17], [], [], [18]]),
    ],
)
def test_cache_batch_sampler(params):
    cache = mock.MagicMock()
    cache.filled = False
    batch_sampler = CacheBatchSampler(params[0], 1, 0, 3, 3, False, True, cache)
    batches = []
    for batch in batch_sampler:
        batches.append(batch)
    assert batches == params[1]
