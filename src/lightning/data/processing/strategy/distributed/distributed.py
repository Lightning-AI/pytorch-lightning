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

from typing import Any, List

import requests
from lightning.data.processing.strategy.base import DataProcessorStrategy
from lightning.data.processing.strategy.distributed.utilities import DistributedMap, HTTPQueue
from lightning.data.streaming.cache import Dir
from lightning.data.utilities.packing import _chunk_list


class MultiNodeDataProcessorStrategy(DataProcessorStrategy):
    def __init__(
        self,
        input_dir: Dir,
        output_dir: Dir,
    ):
        super().__init__()

        self.map = DistributedMap()
        self.queue = HTTPQueue()

        # Ensure the input dir is the same across all nodes
        self.input_dir = self.map.assign("input_dir", input_dir)
        self.output_dir = self.map.assign("output_dir", output_dir)

    def register_inputs(self, inputs: List[Any]) -> None:
        counter = 0
        for items in _chunk_list(inputs, 256):
            for items in items:
                items_dict = {}
                for item in items:
                    items_dict[counter] = item
                    counter += 1

                # Find a better way to handle this
                try:
                    resp = self.queue.put(items_dict)
                    print(resp)
                except requests.exceptions.HTTPError:
                    return

    def get_global_queue(self):
        return self.queue
