import sys
from typing import Any


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
    self = sys.modules[__name__]
    sys.modules["pytorch_lightning.utilities.cli"] = self


class LightningCLI:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`pytorch_lightning.utilities.cli.LightningCLI` was deprecated in v1.7.0 and is no"
            " longer supported as of v1.9.0. Please use `pytorch_lightning.cli.LightningCLI` instead"
        )


class SaveConfigCallback:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`pytorch_lightning.utilities.cli.SaveConfigCallback` was deprecated in v1.7.0 and is no"
            " longer supported as of v1.9.0. Please use `pytorch_lightning.cli.SaveConfigCallback` instead"
        )


class LightningArgumentParser:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`pytorch_lightning.utilities.cli.LightningArgumentParser` was deprecated in v1.7.0 and is no"
            " longer supported as of v1.9.0. Please use `pytorch_lightning.cli.LightningArgumentParser` instead"
        )


def instantiate_class(*_: Any, **__: Any) -> None:
    raise NotImplementedError(
        "`pytorch_lightning.utilities.cli.instantiate_class` was deprecated in v1.7.0 and is no"
        " longer supported as of v1.9.0. Please use `pytorch_lightning.cli.instantiate_class` instead"
    )


_patch_sys_modules()
