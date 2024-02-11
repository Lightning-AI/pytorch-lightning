from lightning.data.utilities.env import _DistributedEnv
from lightning.data.utilities.shuffle import _associate_chunks_and_internals_to_ranks, _intra_node_chunk_shuffle


def test_intra_node_chunk_shuffle():
    chunks_per_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]

    shuffled_indexes = _intra_node_chunk_shuffle(_DistributedEnv(4, 1, 1), chunks_per_ranks, 42, 2)
    assert shuffled_indexes == [5, 2, 0, 7, 6, 1, 3, 4]

    shuffled_indexes = _intra_node_chunk_shuffle(_DistributedEnv(4, 1, 2), chunks_per_ranks, 42, 2)
    assert shuffled_indexes == [3, 2, 1, 0, 7, 6, 5, 4]


    chunks_per_ranks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    shuffled_indexes = _intra_node_chunk_shuffle(_DistributedEnv(8, 7, 2), chunks_per_ranks, 42, 2)
    assert shuffled_indexes == [5, 2, 0, 7, 6, 1, 3, 4, 13, 10, 8, 15, 14, 9, 11, 12]


def test_associate_chunks_and_internals_to_ranks():
    indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    chunk_intervals = [[0, 50], [0, 50], [0, 50], [0, 50], [0, 50], [0, 50], [0, 50], [0, 50]]

    chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert chunks_per_ranks == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert intervals_per_ranks == [[[0, 50], [0, 50]], [[0, 50], [0, 50]], [[0, 50], [0, 50]], [[0, 50], [0, 50]]]

    chunk_intervals = [[0, 50], [0, 150], [0, 50], [0, 12], [0, 50], [0, 27], [0, 50], [0, 33]]

    chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert chunks_per_ranks == [[0, 1], [1, 2], [2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[0]]) == 105
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[1]]) == 105
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[2]]) == 105
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[3]]) == 105
    assert intervals_per_ranks == [
        [[0, 50], [0, 55]],
        [(55, 150), [0, 10]],
        [(10, 50), [0, 12], [0, 50], [0, 3]],
        [(3, 27), [0, 50], [0, 31]],
    ]

    chunk_intervals = [[0, 5], [0, 150], [0, 7], [0, 12], [0, 4], [0, 27], [0, 50], [0, 1]]

    chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert chunks_per_ranks == [[0, 1], [1], [1, 2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[0]]) == 64
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[1]]) == 64
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[2]]) == 64
    assert sum([interval[-1] - interval[0] for interval in intervals_per_ranks[3]]) == 64
    assert intervals_per_ranks == [
        [[0, 5], [0, 59]],
        [[59, 123]],
        [(123, 150), [0, 7], [0, 12], [0, 4], [0, 14]],
        [(14, 27), [0, 50], [0, 1]],
    ]
