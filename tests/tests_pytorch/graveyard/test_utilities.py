import pytest


def test_v2_0_0_rank_zero_from_distributed():
    from pytorch_lightning.utilities.distributed import rank_zero_debug, rank_zero_info

    with pytest.raises(
        RuntimeError, match="rank_zero_debug` was deprecated in v1.6 and is no longer accessible as of v1.8."
    ):
        rank_zero_debug("foo")

    with pytest.raises(
        RuntimeError, match="rank_zero_info` was deprecated in v1.6 and is no longer accessible as of v1.8."
    ):
        rank_zero_info("bar")
