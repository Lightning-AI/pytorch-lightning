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
from typing import Any

from typing_extensions import override

from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.utilities.model_summary import get_human_readable_count


class RichModelSummary(ModelSummary):
    r"""Generates a summary of all layers in a :class:`~lightning.pytorch.core.LightningModule` with `rich text
    formatting <https://github.com/Textualize/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import RichModelSummary

        trainer = Trainer(callbacks=RichModelSummary())

    You could also enable ``RichModelSummary`` using the :class:`~lightning.pytorch.callbacks.RichProgressBar`

    .. code-block:: python

        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import RichProgressBar

        trainer = Trainer(callbacks=RichProgressBar())

    Args:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off.
        **summarize_kwargs: Additional arguments to pass to the `summarize` method.

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.

    """

    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        if not _RICH_AVAILABLE:
            raise ModuleNotFoundError(
                "`RichModelSummary` requires `rich` to be installed. Install it by running `pip install -U rich`."
            )
        super().__init__(max_depth, **summarize_kwargs)

    @staticmethod
    @override
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        from rich import get_console
        from rich.table import Table

        console = get_console()

        header_style: str = summarize_kwargs.get("header_style", "bold magenta")
        table = Table(header_style=header_style)
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")
        table.add_column("Mode")

        column_names = list(zip(*summary_data))[0]

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.print(table)

        parameters = []
        for param in [trainable_parameters, total_parameters - trainable_parameters, total_parameters, model_size]:
            parameters.append("{:<{}}".format(get_human_readable_count(int(param)), 10))

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}")
        grid.add_row(f"[bold]Modules in train mode[/]: {total_training_modes['train']}")
        grid.add_row(f"[bold]Modules in eval mode[/]: {total_training_modes['eval']}")

        console.print(grid)
