import pytest

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


def test_rank_zero_only_raises():
    foo = rank_zero_only(lambda x: x + 1)
    with pytest.raises(RuntimeError, match="rank_zero_only.rank` needs to be set "):
        foo(1)


@pytest.mark.parametrize("rank", [0, 1, 4])
def test_rank_prefixed_message(rank):
    rank_zero_only.rank = rank
    message = rank_prefixed_message("bar", rank)
    assert message == f"[rank: {rank}] bar"
    # reset
    del rank_zero_only.rank


def test_rank_zero_only_default():
    foo = lambda: "foo"
    rank_zero_foo = rank_zero_only(foo, "not foo")

    rank_zero_only.rank = 0
    assert rank_zero_foo() == "foo"

    rank_zero_only.rank = 1
    assert rank_zero_foo() == "not foo"
