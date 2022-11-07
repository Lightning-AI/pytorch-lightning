from typing import Any

import pytorch_lightning as pl


def _rank_zero_debug(*_: Any, **__: Any) -> None:
    # TODO: Remove in v2.0.0
    raise RuntimeError(
        "`pytorch_lightning.utilities.distributed.rank_zero_debug` was deprecated in v1.6 and is no longer accessible"
        " as of v1.8. Use the equivalent function from the `pytorch_lightning.utilities.rank_zero` module instead."
    )


def _rank_zero_info(*_: Any, **__: Any) -> None:
    # TODO: Remove in v2.0.0
    raise RuntimeError(
        "pytorch_lightning.utilities.distributed.rank_zero_info` was deprecated in v1.6 and is no longer accessible"
        " as of v1.8. Use the equivalent function from the `pytorch_lightning.utilities.rank_zero` module instead."
    )


pl.utilities.distributed.rank_zero_debug = _rank_zero_debug
pl.utilities.distributed.rank_zero_info = _rank_zero_info
