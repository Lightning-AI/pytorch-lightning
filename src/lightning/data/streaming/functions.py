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
from pathlib import Path
from types import GeneratorType
from typing import Any, Callable, Optional, Sequence, Union

from lightning.data.streaming.constants import _LIGHTNING_CLOUD_GREATER_EQUAL_0_5_46, _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.streaming.data_processor import DataChunkRecipe, DataProcessor, DataTransformRecipe, PrettyDirectory

if _LIGHTNING_CLOUD_GREATER_EQUAL_0_5_46:
    from lightning_cloud.resolver import _execute, _LightningSrcResolver

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import tree_flatten


def _get_input_dir(inputs: Sequence[Any]) -> str:
    flattened_item, _ = tree_flatten(inputs[0])

    indexed_paths = {
        index: element
        for index, element in enumerate(flattened_item)
        if isinstance(element, str) and os.path.exists(element)
    }

    if len(indexed_paths) == 0:
        raise ValueError(f"The provided item {inputs[0]} didn't contain any filepaths.")

    absolute_path = str(Path(indexed_paths[0]).resolve())

    if indexed_paths[0] != absolute_path:
        raise ValueError("The provided path should be absolute.")

    return "/" + os.path.join(*str(absolute_path).split("/")[:4])


class LambdaDataTransformRecipe(DataTransformRecipe):
    def __init__(self, fn: Callable[[str, Any], None], inputs: Sequence[Any]):
        super().__init__()
        self._fn = fn
        self._inputs = inputs

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, output_dir: str, item_metadata: Any) -> None:  # type: ignore
        self._fn(output_dir, item_metadata)


class LambdaDataChunkRecipe(DataChunkRecipe):
    def __init__(
        self,
        fn: Callable[[Any], None],
        inputs: Sequence[Any],
        chunk_size: Optional[int],
        chunk_bytes: Optional[Union[int, str]],
        compression: Optional[str],
    ):
        super().__init__(chunk_size=chunk_size, chunk_bytes=chunk_bytes, compression=compression)
        self._fn = fn
        self._inputs = inputs

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, item_metadata: Any) -> Any:  # type: ignore
        if isinstance(self._fn, GeneratorType):
            yield from self._fn(item_metadata)
        else:
            yield self._fn(item_metadata)


def map(
    fn: Callable[[str, Any], None],
    inputs: Sequence[Any],
    output_dir: str,
    num_workers: Optional[int] = None,
    fast_dev_run: Union[bool, int] = False,
    num_nodes: Optional[int] = None,
    machine: Optional[str] = None,
    input_dir: Optional[str] = None,
    num_downloaders: int = 1,
) -> None:
    """This function map a callbable over a collection of files possibly in a distributed way.

    Arguments:
        fn: A function to be executed over each input element
        inputs: A sequence of input to be processed by the `fn` function.
            Each input should contain at least a valid filepath.
        output_dir: The folder where the processed data should be stored.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use.
        machine: When doing remote execution, the machine to use.
        num_downloaders: The number of downloaders per worker.

    """
    if not isinstance(inputs, Sequence):
        raise ValueError(f"The provided inputs should be non empty sequence. Found {inputs}.")

    if len(inputs) == 0:
        raise ValueError(f"The provided inputs should be non empty. Found {inputs}.")

    if num_nodes is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        remote_output_dir = _LightningSrcResolver()(output_dir)

        if remote_output_dir is None or "cloudspaces" in remote_output_dir:
            raise ValueError(
                f"The provided `output_dir` isn't valid. Found {output_dir}."
                " HINT: You can either use `/teamspace/s3_connections/...` or `/teamspace/datasets/...`."
            )

        data_processor = DataProcessor(
            num_workers=num_workers or os.cpu_count(),
            remote_output_dir=PrettyDirectory(output_dir, remote_output_dir),
            fast_dev_run=fast_dev_run,
            version=None,
            input_dir=input_dir or _get_input_dir(inputs),
            num_downloaders=num_downloaders,
        )
        return data_processor.run(LambdaDataTransformRecipe(fn, inputs))
    return _execute(
        f"data-prep-map-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_nodes,
        machine,
    )


def optimize(
    fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    output_dir: str,
    chunk_size: Optional[int] = None,
    chunk_bytes: Optional[Union[int, str]] = None,
    compression: Optional[str] = None,
    name: Optional[str] = None,
    num_workers: Optional[int] = None,
    fast_dev_run: bool = False,
    num_nodes: Optional[int] = None,
    machine: Optional[str] = None,
    input_dir: Optional[str] = None,
    num_downloaders: int = 1,
) -> None:
    """This function converts a dataset into chunks possibly in a distributed way.

    Arguments:
        fn: A function to be executed over each input element
        inputs: A sequence of input to be processed by the `fn` function.
            Each input should contain at least a valid filepath.
        output_dir: The folder where the processed data should be stored.
        chunk_size: The maximum number of elements to hold within a chunk.
        chunk_bytes: The maximum number of bytes to hold within a chunk.
        compression: The compression algorithm to use over the chunks.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use.
        machine: When doing remote execution, the machine to use.
        num_downloaders: The number of downloaders per worker.

    """
    if not isinstance(inputs, Sequence):
        raise ValueError(f"The provided inputs should be non empty sequence. Found {inputs}.")

    if len(inputs) == 0:
        raise ValueError(f"The provided inputs should be non empty. Found {inputs}.")

    if chunk_size is None and chunk_bytes is None:
        raise ValueError("Either `chunk_size` or `chunk_bytes` needs to be defined.")

    if num_nodes is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        remote_output_dir = _LightningSrcResolver()(output_dir)

        if remote_output_dir is None or "cloudspaces" in remote_output_dir:
            raise ValueError(
                f"The provided `output_dir` isn't valid. Found {output_dir}."
                " HINT: You can either use `/teamspace/s3_connections/...` or `/teamspace/datasets/...`."
            )

        data_processor = DataProcessor(
            name=name,
            num_workers=num_workers or os.cpu_count(),
            remote_output_dir=PrettyDirectory(output_dir, remote_output_dir),
            fast_dev_run=fast_dev_run,
            input_dir=input_dir or _get_input_dir(inputs),
            num_downloaders=num_downloaders,
        )
        return data_processor.run(
            LambdaDataChunkRecipe(
                fn,
                inputs,
                chunk_size=chunk_size,
                chunk_bytes=chunk_bytes,
                compression=compression,
            )
        )
    return _execute(
        f"data-prep-optimize-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_nodes,
        machine,
    )
