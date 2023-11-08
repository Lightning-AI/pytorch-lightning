# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Any

import lightning.pytorch as pl
from lightning.fabric.strategies import _StrategyRegistry
from lightning.pytorch.accelerators.xla import XLAAccelerator
from lightning.pytorch.plugins.precision import XLAPrecision
from lightning.pytorch.strategies.single_xla import SingleDeviceXLAStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_deprecation


def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules["lightning.pytorch.strategies.single_tpu"] = self
    sys.modules["lightning.pytorch.accelerators.tpu"] = self
    sys.modules["lightning.pytorch.plugins.precision.tpu"] = self
    sys.modules["lightning.pytorch.plugins.precision.tpu_bf16"] = self
    sys.modules["lightning.pytorch.plugins.precision.xlabf16"] = self


class SingleTPUStrategy(SingleDeviceXLAStrategy):
    """Legacy class.

    Use :class:`~lightning.pytorch.strategies.single_xla.SingleDeviceXLAStrategy` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation("The 'single_tpu' strategy is deprecated. Use 'single_xla' instead.")
        super().__init__(*args, **kwargs)

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if "single_tpu" not in strategy_registry:
            strategy_registry.register("single_tpu", cls, description="Legacy class. Use `single_xla` instead.")


class TPUAccelerator(XLAAccelerator):
    """Legacy class.

    Use :class:`~lightning.pytorch.accelerators.xla.XLAAccelerator` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "The `TPUAccelerator` class is deprecated. Use `lightning.pytorch.accelerators.XLAAccelerator` instead."
        )
        super().__init__(*args, **kwargs)


class TPUPrecisionPlugin(XLAPrecision):
    """Legacy class.

    Use :class:`~lightning.pytorch.plugins.precision.xla.XLAPrecision` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "The `TPUPrecisionPlugin` class is deprecated. Use `lightning.pytorch.plugins.precision.XLAPrecision`"
            " instead."
        )
        super().__init__(precision="32-true")


class TPUBf16PrecisionPlugin(XLAPrecision):
    """Legacy class.

    Use :class:`~lightning.pytorch.plugins.precision.xlabf16.XLAPrecision` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "The `TPUBf16PrecisionPlugin` class is deprecated. Use"
            " `lightning.pytorch.plugins.precision.XLAPrecision` instead."
        )
        super().__init__(precision="bf16-true")


class XLABf16PrecisionPlugin(XLAPrecision):
    """Legacy class.

    Use :class:`~lightning.pytorch.plugins.precision.xlabf16.XLAPrecision` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "The `XLABf16PrecisionPlugin` class is deprecated. Use"
            " `lightning.pytorch.plugins.precision.XLAPrecision` instead."
        )
        super().__init__(precision="bf16-true")


def _patch_classes() -> None:
    setattr(pl.strategies, "SingleTPUStrategy", SingleTPUStrategy)
    setattr(pl.accelerators, "TPUAccelerator", TPUAccelerator)
    setattr(pl.plugins, "TPUPrecisionPlugin", TPUPrecisionPlugin)
    setattr(pl.plugins.precision, "TPUPrecisionPlugin", TPUPrecisionPlugin)
    setattr(pl.plugins, "TPUBf16PrecisionPlugin", TPUBf16PrecisionPlugin)
    setattr(pl.plugins.precision, "TPUBf16PrecisionPlugin", TPUBf16PrecisionPlugin)
    setattr(pl.plugins, "XLABf16PrecisionPlugin", XLABf16PrecisionPlugin)
    setattr(pl.plugins.precision, "XLABf16PrecisionPlugin", XLABf16PrecisionPlugin)


_patch_sys_modules()
_patch_classes()

SingleTPUStrategy.register_strategies(pl.strategies.StrategyRegistry)  # type: ignore[has-type]
