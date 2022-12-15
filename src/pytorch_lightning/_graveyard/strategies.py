from typing import Any

import pytorch_lightning.strategies.deepspeed as deepspeed


class _LightningDeepSpeedModule:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.strategies.deepspeed.LightningDeepSpeedModule` was deprecated in v1.7.1 and is no"
            " longer supported as of v1.9.0."
        )


deepspeed.LightningDeepSpeedModule = _LightningDeepSpeedModule
