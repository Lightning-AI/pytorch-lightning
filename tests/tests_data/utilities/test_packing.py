import pytest
from lightning.data.utilities.packing import _pack_greedily


def test_pack_greedily():
    with pytest.raises(ValueError, match="must have the same length"):
        _pack_greedily(items=["A"], weights=[], num_bins=1)
    with pytest.raises(ValueError, match="must have the same length"):
        _pack_greedily(items=[], weights=[1], num_bins=1)
    with pytest.raises(ValueError, match="must be positive"):
        _pack_greedily(items=["A"], weights=[0], num_bins=1)
    with pytest.raises(ValueError, match="must be positive"):
        _pack_greedily(items=["A"], weights=[-1], num_bins=1)

    assert _pack_greedily(items=[], weights=[], num_bins=0) == ({}, {})
    assert _pack_greedily(items=[], weights=[], num_bins=1) == ({}, {0: 0})

    # one item, one bin
    bin_contents, bin_weights = _pack_greedily(items=["A"], weights=[1], num_bins=1)
    assert bin_contents == {0: ["A"]}
    assert bin_weights == {0: 1}

    # more bins than items
    bin_contents, bin_weights = _pack_greedily(items=["A"], weights=[1], num_bins=3)
    assert bin_contents == {0: ["A"]}
    assert bin_weights == {0: 1, 1: 0, 2: 0}

    # items with equal weight
    bin_contents, bin_weights = _pack_greedily(items=["A", "B", "C", "D"], weights=[3, 3, 3, 3], num_bins=4)
    assert bin_contents == {0: ["A"], 1: ["B"], 2: ["C"], 3: ["D"]}
    assert bin_weights == {0: 3, 1: 3, 2: 3, 3: 3}

    # pigeonhole principle: more items than bins
    bin_contents, bin_weights = _pack_greedily(items=["A", "B", "C", "D"], weights=[1, 1, 1, 1], num_bins=3)
    assert bin_contents == {0: ["A", "D"], 1: ["B"], 2: ["C"]}
    assert bin_weights == {0: 2, 1: 1, 2: 1}

    bin_contents, bin_weights = _pack_greedily(
        items=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        weights=[4, 1, 2, 5, 8, 7, 3, 6, 9],
        num_bins=3,
    )
    assert bin_contents == {0: ["I", "A", "G"], 1: ["E", "D", "C"], 2: ["F", "H", "B"]}
    assert bin_weights == {0: 16, 1: 15, 2: 14}
