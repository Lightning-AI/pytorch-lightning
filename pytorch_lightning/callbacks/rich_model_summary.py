# Copyright The PyTorch Lightning team.
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
from typing import List, Union

from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from pytorch_lightning.utilities.model_summary import get_human_readable_count

if _RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Table


class RichModelSummary(ModelSummary):
    @staticmethod
    def summarize(
        summary_data: List[List[Union[str, List[str]]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
    ) -> None:

        console = Console()

        table = Table(title="Model Summary")

        table.add_column(" ")
        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Params", justify="right", style="green")

        # print(summary_data)
        # if self._model.example_input_array is not None:
        #     table.add_column("In sizes", justify="right", style="green")
        #     table.add_column("Out sizes", justify="right", style="green")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.print(table)

        # Formatting
        s = "{:<{}}"

        parameters = []
        for param in [trainable_parameters, total_parameters - trainable_parameters, total_parameters, model_size]:
            parameters.append(s.format(get_human_readable_count(param), 10))

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}")

        console.print(grid)
