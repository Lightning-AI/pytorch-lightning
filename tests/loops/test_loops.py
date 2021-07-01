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

from typing import Dict, Iterator

from pytorch_lightning.loops.base import Loop


def test_loop_restore():

    class CustomExpection(Exception):
        pass

    class Simple(Loop):

        def __init__(self, dataset: Iterator):
            super().__init__()
            self.dataset = dataset

        def restore(self) -> None:
            self.iter_dataset = iter(self.dataset)
            for _ in range(self.iteration_count):
                next(self.iter_dataset)
            self.iteration_count += 1

        @property
        def done(self) -> bool:
            return self.iteration_count > len(self.dataset)

        def reset(self) -> None:
            self.iter_dataset = iter(self.dataset)
            self.outputs = []

        def advance(self) -> None:
            value = next(self.iter_dataset)
            print(value, self.iteration_count)

            if self.iteration_count == 5:
                raise CustomExpection

            self.outputs.append(value)

        def state_dict(self) -> Dict:
            return {"iteration_count": self.iteration_count, "outputs": self.outputs}

        def load_state_dict(self, state_dict: Dict) -> None:
            self.iteration_count = state_dict["iteration_count"]
            self.outputs = state_dict["outputs"]

    state_dict = {}

    data = range(10)
    loop = Simple(data)
    try:
        loop.run()
    except CustomExpection:
        state_dict = loop.state_dict()

    loop = Simple(data)
    loop.load_state_dict(state_dict)
    loop.is_restarting = True
    loop.run()

    assert loop.outputs == list(range(10))
