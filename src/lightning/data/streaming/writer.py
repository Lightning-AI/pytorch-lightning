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

import json
import os
from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv
from lightning.data.streaming.compression import _COMPRESSORS, Compressor
from lightning.data.streaming.constants import _INDEX_FILENAME, _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.streaming.serializers import _SERIALIZERS, Serializer

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import PyTree, tree_flatten, treespec_dumps


def _get_data_optimizer_node_rank() -> Optional[int]:
    node_rank = os.getenv("DATA_OPTIMIZER_NODE_RANK", None)
    if node_rank is not None:
        return int(node_rank)
    return node_rank


@dataclass
class Item:
    index: int
    data: bytes
    bytes: int
    dim: Optional[int] = None

    def __len__(self) -> int:
        return self.bytes


class BinaryWriter:
    def __init__(
        self,
        cache_dir: str,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
        compression: Optional[str] = None,
        follow_tensor_dimension: bool = True,
    ):
        """The BinaryWriter enables to chunk dataset into an efficient streaming format for cloud training.

        Arguments:
            cache_dir: The path to where the chunks will be saved.
            chunk_bytes: The maximum number of bytes within a chunk.
            chunk_size: The maximum number of items within a chunk.
            compression: The compression algorithm to use.

        """
        self._cache_dir = cache_dir

        if (isinstance(self._cache_dir, str) and not os.path.exists(self._cache_dir)) or self._cache_dir is None:
            raise FileNotFoundError(f"The provided cache directory `{self._cache_dir}` doesn't exist.")

        if (chunk_size is None and chunk_bytes is None) or (chunk_size and chunk_bytes):
            raise ValueError("Either one of the `chunk_size` or the `chunk_bytes` need to be provided.")

        self._serializers: Dict[str, Serializer] = _SERIALIZERS
        self._chunk_size = chunk_size
        self._chunk_bytes = chunk_bytes
        self._compression = compression

        self._data_format: Optional[List[str]] = None
        self._data_spec: Optional[PyTree] = None

        if self._compression:
            if len(_COMPRESSORS) == 0:
                raise ValueError("No compresion algorithms are installed.")
            if self._compression not in _COMPRESSORS:
                raise ValueError(
                    f"The provided compression {self._compression} isn't available in {sorted(_COMPRESSORS)}"
                )
            self._compressor: Compressor = _COMPRESSORS[self._compression]

        self._serialized_items: Dict[int, Item] = {}
        self._chunk_index = 0
        self._min_index: Optional[int] = None
        self._max_index: Optional[int] = None
        self._chunks_info: List[Dict[str, Any]] = []
        self._worker_env: Optional[_WorkerEnv] = None
        self._rank: Optional[int] = None
        self._is_done = False
        self._distributed_env = _DistributedEnv.detect()
        self._follow_tensor_dimension = follow_tensor_dimension

    @property
    def filled(self) -> bool:
        """Returns whether the caching phase is done."""
        if self._is_done:
            return True
        files = os.listdir(self._cache_dir)
        index_files = [f for f in files if f.endswith(_INDEX_FILENAME)]
        worker_end = _WorkerEnv.detect()
        data_optimiser_num_workers = os.getenv("DATA_OPTIMIZER_NUM_WORKERS", None)
        if data_optimiser_num_workers is not None:
            self._is_done = len(index_files) == int(data_optimiser_num_workers)
        else:
            self._is_done = len(index_files) == self._distributed_env.world_size * worker_end.world_size
        return self._is_done

    @property
    def rank(self) -> int:
        """Returns the rank of the writer."""
        if self._rank is None:
            rank = os.getenv("DATA_OPTIMIZER_GLOBAL_RANK", None)
            if rank:
                self._rank = int(rank)
            else:
                self._worker_env = _WorkerEnv.detect()
                self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the writer."""
        out = {
            "compression": self._compression,
            "chunk_size": self._chunk_size,
            "chunk_bytes": self._chunk_bytes,
            "data_format": self._data_format,
            "data_spec": treespec_dumps(self._data_spec) if self._data_spec else None,
        }
        return out

    def serialize(self, items: Any) -> Tuple[bytes, Optional[int]]:
        """Serialize a dictionary into its binary format."""

        # Flatten the items provided by the users
        flattened, data_spec = tree_flatten(items)

        is_single_tensor = len(flattened) == 1 and isinstance(flattened[0], torch.Tensor)

        # Collect the sizes and associated bytes for each item
        sizes: List[int] = []
        data: List[bytes] = []

        data_format: List[str] = []
        for item in flattened:
            data_format.append(self._serialize(item, sizes, data))

        if self._data_format is None:
            self._data_format = data_format
        elif self._data_format != data_format and self._should_raise(data_format, self._data_format):
            raise ValueError(
                f"The data format changed between items. Found {data_format} instead of {self._data_format}."
            )

        if self._data_spec is None:
            self._data_spec = data_spec
        elif self._data_spec != data_spec:
            raise Exception(f"The data format changed between items. Found {data_spec} instead of {self._data_spec}.")

        # If there is a single element and it is a tensor, enable continous array.
        if is_single_tensor:
            return data[0], flattened[0].shape[0]

        # Concatenante into a single byte array
        head = np.array(sizes, np.uint32).tobytes()
        body = b"".join(data)
        return head + body, None

    def _serialize(self, item: Any, sizes: List[int], data: List[bytes]) -> str:
        """Serialize a given item and append its size and bytes to the sizes and data array."""
        for serializer_name, serializer in self._serializers.items():
            if serializer.can_serialize(item):
                serialized_item, name = serializer.serialize(item)
                data.append(serialized_item)
                sizes.append(len(serialized_item))
                return name or serializer_name
        raise ValueError(f"The provided item isn't serializable. Found {item}")

    def _create_chunk(self, filename: str, on_done: bool = False) -> bytes:
        """Create a binary chunk from all the binarized items."""
        if on_done:
            indices = sorted(self._serialized_items.keys())
            for i in range(len(indices) - 1):
                assert indices[i] == indices[i + 1] - 1, indices
            min_index = indices[0]
            max_index = indices[-1] + 1
            num_items = np.uint32(max_index - min_index)
            items = [self._serialized_items.pop(index) for index in indices]
        else:
            assert self._max_index is not None, (self._max_index, self._min_index)
            assert self._min_index is not None, (self._max_index, self._min_index)
            num_items = np.uint32(self._max_index - self._min_index)
            items = [self._serialized_items.pop(index) for index in range(self._min_index, self._max_index)]
            min_index = self._min_index
            max_index = self._max_index

        if len(items) == 0:
            raise RuntimeError("The items shouldn't have an empty length. Something went wrong.")

        sizes = list(map(len, items))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        offsets += len(num_items.tobytes()) + len(offsets.tobytes())
        sample_data = b"".join([item.data for item in items])
        data = num_items.tobytes() + offsets.tobytes() + sample_data
        offsets = offsets.tolist()

        current_chunk_bytes = sum([item.bytes for item in items])

        if self._chunk_bytes:
            assert current_chunk_bytes <= self._chunk_bytes

        if self._chunk_size:
            assert num_items.item() <= self._chunk_size

        dim: Optional[int] = None
        if items[0].dim:
            dim = sum([item.dim if item.dim is not None else 0 for item in items])

        chunk_info = {
            "chunk_bytes": current_chunk_bytes,
            "chunk_size": num_items.item(),
            "filename": filename,
            "dim": dim,
        }

        self._chunks_info.append(chunk_info)

        return data

    def get_chunk_filename(self) -> str:
        if self._compression:
            return f"chunk-{self.rank}-{self._chunk_index}.{self._compression}.bin"
        return f"chunk-{self.rank}-{self._chunk_index}.bin"

    def write_chunk(self, on_done: bool = False) -> str:
        """Write a chunk to the filesystem."""
        filename = self.get_chunk_filename()
        self.write_chunk_to_file(self._create_chunk(filename, on_done=on_done), filename)
        self._chunk_index += 1
        return os.path.join(self._cache_dir, filename)

    def __setitem__(self, index: int, items: Any) -> None:
        """Store an item to a chunk.

        The index needs to be provided in order.

        This is handled by the samplers automatically. This ensures we can map an index to a shard from an interval.

        """
        self.add_item(index, items)

    def add_item(self, index: int, items: Any) -> Optional[str]:
        # Track the minimum index provided to the writer
        # Serialize the items and store an Item object.
        if index in self._serialized_items:
            raise ValueError(f"The provided index {index} already exists in the cache.")

        data, dim = self.serialize(items)
        self._serialized_items[index] = Item(
            index=index,
            data=data,
            bytes=len(data),
            dim=dim,
        )

        if self._should_write():
            filepath = os.path.join(self._cache_dir, self.get_chunk_filename())
            self.write_chunk()
            self._min_index = None
            self._max_index = None
            return filepath

    def _should_write(self) -> bool:
        if not self._serialized_items:
            return False
        indexes = list(self._serialized_items.keys())
        self._min_index = index = indexes[0] if len(indexes) == 1 else min(*indexes)
        num_bytes = 0
        num_items = 0
        while True:
            item = self._serialized_items.get(index, None)
            if item:
                num_bytes += item.bytes
                num_items += item.dim if item.dim else 1
                index += 1
                if (self._chunk_bytes and self._chunk_bytes < num_bytes) or (
                    self._chunk_size and num_items > self._chunk_size
                ):
                    self._max_index = index - 1
                    return True
            else:
                return False

    def write_chunk_to_file(
        self,
        raw_data: bytes,
        filename: str,
    ) -> None:
        """Write chunk bytes to a file."""
        # Whether to compress the raw bytes
        if self._compression:
            raw_data = self._compressor.compress(raw_data)

        # Write the binary chunk file
        with open(os.path.join(self._cache_dir, filename), "wb") as out:
            out.write(raw_data)

    def write_chunks_index(self) -> str:
        """Write the chunks index to a JSON file."""
        filepath = os.path.join(self._cache_dir, f"{self.rank}.{_INDEX_FILENAME}")
        config = self.get_config()
        with open(filepath, "w") as out:
            json.dump({"chunks": self._chunks_info, "config": config}, out, sort_keys=True)
        return filepath

    def done(self) -> List[str]:
        """Called when StopIteration is triggered."""
        filepaths: List[str] = []
        if self.filled:
            return filepaths

        # Try writing down an chunks
        while self._should_write():
            filepaths.append(self.write_chunk())

        # If any elements is left, try writing one last chunk
        if self._serialized_items:
            filepaths.append(self.write_chunk(True))

        # Write down the index file
        self.write_chunks_index()

        self._is_done = True
        return filepaths

    def merge(self, num_workers: int = 1, node_rank: Optional[int] = None) -> None:
        """Once all the workers have written their own index, the merge function is responsible to read and merge them
        into a single index."""
        num_workers = num_workers or 1

        # Only for non rank 0
        if self.rank != 0:
            while not os.path.exists(os.path.join(self._cache_dir, _INDEX_FILENAME)):
                sleep(0.001)
            return

        # Wait for all indexes to be available
        is_done = False
        while not is_done:
            files = os.listdir(self._cache_dir)

            # Return if the index already exists
            if _INDEX_FILENAME in files:
                return

            index_files = [f for f in files if f.endswith(_INDEX_FILENAME)]

            # When using the Data Optimizer, we don't use multi processes.
            is_done = len(index_files) == self._distributed_env.world_size * num_workers
            sleep(0.001)

        self._merge_no_wait(node_rank=node_rank)

    def _merge_no_wait(self, node_rank: Optional[int] = None) -> None:
        """Once all the workers have written their own index, the merge function is responsible to read and merge them
        into a single index."""
        files = os.listdir(self._cache_dir)
        index_files = [f for f in files if f.endswith(_INDEX_FILENAME)]

        chunks_info = []
        config = None
        for index_filename in sorted(index_files):
            chunk_path = os.path.join(self._cache_dir, index_filename)
            with open(chunk_path) as f:
                data = json.load(f)

                if config is None:
                    config = data["config"]

                elif config != data["config"]:
                    breakpoint()
                    raise Exception("The config isn't consistent between chunks. This shouldn't have happened.")

                chunks_info.extend(data["chunks"])

            os.remove(chunk_path)

        if node_rank is None:
            with open(os.path.join(self._cache_dir, _INDEX_FILENAME), "w") as f:
                json.dump({"chunks": chunks_info, "config": config}, f, sort_keys=True)
        else:
            with open(os.path.join(self._cache_dir, f"{node_rank}-{_INDEX_FILENAME}"), "w") as f:
                json.dump({"chunks": chunks_info, "config": config}, f, sort_keys=True)

    def _should_raise(self, data_format_1: List[str], data_format_2: List[str]) -> bool:
        if len(data_format_1) != len(data_format_2):
            return True

        def is_non_valid(f1: str, f2: str) -> bool:
            if f1 in ["pil", "jpeg"] and f2 in ["pil", "jpeg"]:
                return False
            return f1 != f2

        return any(is_non_valid(f1, f2) for f1, f2 in zip(data_format_1, data_format_2))
