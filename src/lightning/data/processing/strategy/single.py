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

from multiprocessing import Queue
from typing import Any, List

from lightning.data.processing.strategy.base import DataProcessorStrategy
from lightning.data.streaming.cache import Dir


class SingleNodeDataProcessorStrategy(DataProcessorStrategy):
    def __init__(self, input_dir: Dir, output_dir: Dir):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.queue = Queue()

        print(f"Storing the files under {self.output_dir.path}")

    def register_inputs(self, inputs: List[Any]) -> None:
        for k, v in enumerate(inputs):
            self.queue.put({k: v})

        # Indicate end of processing
        self.queue.put(None)

    def get_global_queue(self):
        return self.queue
