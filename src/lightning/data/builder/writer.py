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
from typing import Any, Dict, Optional

import numpy as np
from lightning_utilities.core.imports import RequirementCache

from lightning.data.builder.base import BaseWriter
from lightning.data.builder.serializers import _SERIALIZERS

_PIL_AVAILABLE = RequirementCache("PIL")

if _PIL_AVAILABLE:
    from PIL import Image
else:
    Image = Any


class Writer(BaseWriter):
    def __init__(
        self,
        out_dir: str,
        dict_format: Dict[str, str],
        chunk_size: int = 1 << 26,
        compression: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(out_dir, chunk_size, compression, name)

        if not _PIL_AVAILABLE:
            raise Exception("The ImageWriter requires pil to be installed")

        self._dict_format = {k.lower(): v for k, v in dict_format.items()}
        self._dict_format_keys = sorted(self._dict_format.keys())
        self._serializers = _SERIALIZERS

        available_serializers = set(self._serializers.keys())
        selected_serializers = set(self._dict_format.values())
        if selected_serializers.difference(available_serializers):
            raise Exception(
                f"The provided dict_format don't match the provided serializers. Should be selected from {available_serializers}."
            )

        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        self._config_data = text.encode("utf-8")

    def get_config(self) -> Dict[str, Any]:
        out = super().get_config()
        out.update(self._dict_format)
        return out

    def serialize(self, items: Dict[str, Any]) -> bytes:
        if not isinstance(items, dict):
            raise Exception("The provided data should be a dictionary.")

        keys = sorted(items.keys())

        if keys != self._dict_format_keys:
            raise Exception(
                f"The provided keys don't match the provided format. Found {keys} instead of {self._dict_format_keys}."
            )

        sizes = []
        data = []

        for key in self._dict_format_keys:
            serializer_name = self._dict_format[key]
            serializer = self._serializers[serializer_name]
            serialized_data = serializer.serialize(items[key])

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

        self._chunks.append(
            {
                "samples": len(self._serialized_items),
                "config": self.get_config(),
                "filename": filename,
            }
        )

        return num_items.tobytes() + offsets.tobytes() + self._config_data + sample_data

    def write_chunk(self, rank: int):
        filename = f"chunk-{rank}-{self._counter}.bin"
        self.write_file(self._create_chunk(filename), filename)

    def reset(self):
        pass
