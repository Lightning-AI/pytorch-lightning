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

from typing import Any, Literal, Optional, Union

from torch.utils.data import Dataset

from lightning.data.streaming import Cache


class StreamingDataset(Dataset):
    """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class."""

    def __init__(
        self, name: str, version: Optional[Union[int, Literal["latest"]]] = "latest", cache_dir: Optional[str] = None
    ) -> None:
        """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class.

        Arguments:
            name: The name of the optimised dataset.
            version: The version of the dataset to use.
            cache_dir: The cache dir where the data would be stored.

        """
        super().__init__()
        self.cache = Cache(name=name, version=version, cache_dir=cache_dir)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> Any:
        return self.cache[idx]

    def getitem(self, obj: Any) -> Any:
        """Override the getitem with your own logic to transform the cache object."""
        return obj
