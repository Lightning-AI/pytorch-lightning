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
from typing import TYPE_CHECKING, Any, Callable, Union

from typing_extensions import override

from lightning.fabric.strategies.launchers.launcher import _Launcher
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

if TYPE_CHECKING:
    from lightning.fabric.strategies import XLAFSDPStrategy, XLAStrategy


class _XLALauncher(_Launcher):
    r"""Launches processes that run a given function in parallel on XLA supported hardware, and joins them all at the
    end.

    The main process in which this launcher is invoked creates N so-called worker processes (using the
    `torch_xla` :func:`xmp.spawn`) that run the given function.
    Worker processes have a rank that ranges from 0 to N - 1.

    Note:
        - This launcher requires all objects to be pickleable.
        - It is important that the entry point to the program/script is guarded by ``if __name__ == "__main__"``.

    Args:
        strategy: A reference to the strategy that is used together with this launcher

    """

    def __init__(self, strategy: Union["XLAStrategy", "XLAFSDPStrategy"]) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.strategies.xla.launcher import XLALauncherFabric as EnterpriseXLALauncher

        self.xla_impl = EnterpriseXLALauncher(strategy=strategy)

    @property
    @override
    def is_interactive_compatible(self) -> bool:
        return self.xla_impl.is_interactive_compatible

    @override
    def launch(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            **kwargs: Optional keyword arguments to be passed to the given function.

        """
        return self.xla_impl.launch(function=function, *args, **kwargs)

    @property
    def _start_method(self) -> str:
        return self.xla_impl._start_method

    @_start_method.setter
    def _start_method(self, start_method: str) -> None:
        self.xla_impl._start_method = start_method
