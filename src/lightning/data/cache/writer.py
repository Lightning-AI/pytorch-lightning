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
from typing import Any, Dict, Optional

import numpy as np

from lightning.data.cache.compression import _COMPRESSORS
from lightning.data.cache.env import _WorkerEnv
from lightning.data.cache.serializers import _SERIALIZERS
from lightning.data.datasets.env import _DistributedEnv


def cloud_path(cache_dir: str) -> Optional[str]:
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    cloud_space_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)

    if cluster_id is None or project_id is None or cloud_space_id is None:
        return None
    return f"s3://{cluster_id}/projects/{project_id}/cloudspaces/{cloud_space_id}/content/{cache_dir}/"


class BinaryWriter:
    def __init__(
        self,
        cache_dir: str,
        data_format: Dict[str, str],
        chunk_size: int = 1 << 26,
        compression: Optional[str] = None,
    ):
        """The BinaryWriter enables to chunk dataset into an efficient streaming format for cloud training.

        Arguments:
            cache_dir: The path to where the chunks will be saved.
            data_format: The format of the provided data to cache. Only dictionary are supported for now.
            chunk_size: The maximum number of bytes to store within a chunk.
            compression: The compression algorithm to use.

        """
        self._cache_dir = cache_dir
        self._data_format = {k.lower(): v for k, v in data_format.items()}
        self._chunk_size = chunk_size
        self._compression = compression

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache directory `{self._cache_dir}` doesn't exist.")

        if len(self._data_format) == 0:
            raise ValueError("The provided data format shouldn't be empty.")

        self._data_format_keys = sorted(self._data_format.keys())
        self._serializers = _SERIALIZERS

        available_serializers = set(self._serializers.keys())
        selected_serializers = set(self._data_format.values())
        if selected_serializers.difference(available_serializers):
            raise ValueError(
                "The provided data_format don't match the provided serializers."
                " Should be selected from {sorted(available_serializers)}."
            )

        if self._compression:
            if len(_COMPRESSORS) == 0:
                raise ValueError("No compresion algorithms are installed.")
            if self._compression not in _COMPRESSORS:
                raise ValueError(
                    f"The provided compression {self._compression} isn't available in {sorted(_COMPRESSORS)}"
                )
            self._compressor = _COMPRESSORS[self._compression]

        self._current_chunk_size = 0
        self._chunk_id = 0
        self._serialized_items = []
        self._chunks_info = []
        self._indexes = []
        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        self._config_data = text.encode("utf-8")

        self._env = _DistributedEnv.detect()
        self._worker_env = None
        self._rank = None
        self._is_done = False

    @property
    def rank(self):
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    def get_config(self) -> Dict[str, Any]:
        out = super().get_config()
        out.update(self._data_format)

        cloud_path = self.get_cloud_path(self._cache_dir)
        if cloud_path:
            out["cloud_path"] = cloud_path
        return out

    def serialize(self, items: Dict[str, Any]) -> bytes:
        if not isinstance(items, dict):
            raise Exception("The provided data should be a dictionary.")

        keys = sorted(items.keys())

        if keys != self._data_format_keys:
            raise Exception(
                f"The provided keys don't match the provided format. Found {keys} instead of {self._data_format_keys}."
            )

        sizes = []
        data = []

        for key in self._data_format_keys:
            serializer_name = self._data_format[key]
            serializer = self._serializers[serializer_name]
            serialized_data = serializer.serialize(items[key]) if not isinstance(items[key], bytes) else items[key]
            sizes.append(len(serialized_data))
            data.append(serialized_data)

        head = np.array(sizes, np.uint32).tobytes()
        body = b"".join(data)
        return head + body

    def _create_chunk(self, filename: str) -> bytes:
        num_items = np.uint32(len(self._serialized_items))
        sizes = list(map(len, self._serialized_items))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        offsets += len(num_items.tobytes()) + len(offsets.tobytes()) + len(self._config_data)
        sample_data = b"".join(self._serialized_items)
        data = num_items.tobytes() + offsets.tobytes() + self._config_data + sample_data
        offsets = offsets.tolist()
        mapping = {}
        for i in range(len(self._indexes)):
            mapping[self._indexes[i]] = [offsets[i], offsets[i + 1]]

        assert len(mapping) == len(self._indexes)

        chunk_info = {
            "samples": len(self._serialized_items),
            "config": self.get_config(),
            "filename": filename,
            "mapping": mapping,
        }

        self._chunks_info.append(chunk_info)

        return data

    def write_chunk(self):
        if self._compression:
            filename = f"chunk-{self.rank}-{self._chunk_id}.{self._compression}.bin"
        else:
            filename = f"chunk-{self.rank}-{self._chunk_id}.bin"
        self.write_file(self._create_chunk(filename), filename)

    @property
    def is_cached(self) -> bool:
        return os.path.exists(os.path.join(self._cache_dir, "index.json"))

    def get_config(self) -> Dict[str, Any]:
        return {"compression": self._compression, "chunk_size": self._chunk_size, "data_format": self._data_format}

    @property
    def available_serializers(self):
        return self._serializers

    def reset(self) -> None:
        """Reset the writer to handle the next chunk."""
        self._serialized_items = []
        self._indexes = []
        self._current_chunk_size = 0

    def __setitem__(self, index, items: any):
        serialized_items = self.serialize(items)
        serialized_items_size = len(serialized_items)

        if self._chunk_size < self._current_chunk_size + serialized_items_size:
            if self._current_chunk_size == 0:
                raise Exception(
                    f"The provided chunk_size {self._chunk_size} is too small."
                    f" You should use a multiple of {serialized_items_size} bytes."
                )
            self.write_chunk()
            self.reset()
            self._chunk_id += 1

        self._serialized_items.append(serialized_items)
        self._current_chunk_size += serialized_items_size

        if self._indexes:
            assert self._indexes[-1] == index - 1

        self._indexes.append(index)

    def write_file(
        self,
        raw_data: bytes,
        filename: str,
    ) -> None:
        if self._compression:
            raw_data = self._compressor.compress(raw_data)
        filepath = os.path.join(self._cache_dir, filename)
        with open(filepath, "wb") as out:
            out.write(raw_data)

    def write_chunks_index(self):
        filepath = os.path.join(self._cache_dir, f"{self.rank}.index.json")
        with open(filepath, "w") as out:
            json.dump({"chunks": self._chunks_info}, out, sort_keys=True)

    def done(self):
        if self._is_done:
            return
        if self._serialized_items:
            self.write_chunk()
        self.write_chunks_index()
        self.reset()
        self._is_done = True

    @classmethod
    def get_cloud_path(cls, cache_dir: str) -> Optional[str]:
        cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
        project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
        cloud_space_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)

        if cluster_id is None or project_id is None or cloud_space_id is None:
            return None
        cache_dir = cache_dir.replace("~/", "").replace("~", "").replace("/teamspace/studios/this_studio/", "")
        if cache_dir.startswith("/"):
            cache_dir = cache_dir[1:]
        return os.path.join(
            f"s3://{cluster_id}/projects/{project_id}/cloudspaces/{cloud_space_id}/code/content", cache_dir
        )
