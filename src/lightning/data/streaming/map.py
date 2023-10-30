# Copyright The Lightning AI team.
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

import os
from datetime import datetime
from typing import Any, Callable, List, Optional

from lightning.data.streaming.data_processor import DataProcessor, DataTransformRecipe


class LambdaDataTransformRecipe(DataTransformRecipe):
    def __init__(self, fn: Callable[[str, Any], None], inputs: List[Any]):
        super().__init__()
        self._fn = fn
        self._inputs = inputs

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, output_dir: str, item_metadata: Any) -> None:  # type: ignore
        self._fn(output_dir, item_metadata)


def map(
    fn: Callable[[str, Any], None],
    inputs: Any,
    num_workers: Optional[int] = None,
    name: Optional[str] = None,
    remote_output_dir: Optional[str] = None,
    fast_dev_run: bool = False,
) -> None:
    """This function executes a function over a collection of files possibly in a distributed way."""

    data_processor = DataProcessor(
        name=name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        num_workers=num_workers or os.cpu_count(),
        remote_output_dir=remote_output_dir,
        fast_dev_run=fast_dev_run,
    )
    data_processor.run(LambdaDataTransformRecipe(fn, inputs))
