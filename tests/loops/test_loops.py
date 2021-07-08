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
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterator

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.progress import BaseProgress


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

            if self.iteration_count == 5:
                raise CustomExpection

            self.outputs.append(value)

        def state_dict(self) -> Dict:
            return {"iteration_count": self.iteration_count, "outputs": self.outputs}

        def load_state_dict(self, state_dict: Dict) -> None:
            self.iteration_count = state_dict["iteration_count"]
            self.outputs = state_dict["outputs"]

    data = range(10)
    loop = Simple(data)
    try:
        loop.run()
        state_dict = {}
    except CustomExpection:
        state_dict = loop.state_dict()

    loop = Simple(data)
    loop.load_state_dict(state_dict)
    loop.restarting = True
    loop.run()

    assert not loop.restarting
    assert loop.outputs == list(range(10))


def test_loop_hierarchy():

    @dataclass
    class SimpleProgress(BaseProgress):

        increment: int = 0

        def state_dict(self):
            return {"increment": self.increment}

        def load_state_dict(self, state_dict):
            self.increment = state_dict["increment"]

    class Simple(Loop):

        def __init__(self, a):
            super().__init__()
            self.a = a
            self.progress = SimpleProgress()

        def advance(self, *args: Any, **kwargs: Any) -> None:
            for loop in self._loops.values():
                loop.run()
                self.progress.increment += 1
            self.progress.increment += 1

        @property
        def done(self) -> bool:
            return self.iteration_count > 0

        def reset(self) -> None:
            pass

        def restore(self) -> None:
            pass

        def state_dict(self) -> Dict:
            return {"a": self.a}

        def load_state_dict(self, state_dict: Dict) -> None:
            self.a = state_dict["a"]

    loop_parent = Simple(1)
    loop_child = Simple(2)
    loop_parent.loop_child = loop_child
    state_dict = loop_parent.get_state_dict()
    assert state_dict == OrderedDict([('state_dict', {
        'a': 1
    }), ('progress', {
        'increment': 0
    }), ('loop_child.state_dict', {
        'a': 2
    }), ('loop_child.progress', {
        'increment': 0
    })])

    state_dict["loop_child.state_dict"]["a"] = 3
    loop_parent._load_state_dict(state_dict)
    assert loop_parent.restarting

    loop_parent.run()

    loop_parent_copy = deepcopy(loop_parent)
    assert loop_parent_copy.get_state_dict() == loop_parent.get_state_dict()

    assert loop_parent_copy.state_dict() == {'a': 1}
    assert loop_parent_copy.loop_child.state_dict() == {'a': 3}

    assert not loop_parent.restarting

    state_dict = loop_parent.get_state_dict()
    assert state_dict == OrderedDict([('state_dict', {
        'a': 1
    }), ('progress', {
        'increment': 2
    }), ('loop_child.state_dict', {
        'a': 3
    }), ('loop_child.progress', {
        'increment': 1
    })])

    loop_parent = Simple(1)
    loop_child = Simple(2)
    loop_parent.loop_child = loop_child
    loop_parent._load_state_dict(state_dict)
    assert loop_parent.progress.increment == 2
    assert loop_parent.loop_child.progress.increment == 1

    del loop_parent.loop_child
    state_dict = loop_parent.get_state_dict()
    assert state_dict == OrderedDict([('state_dict', {'a': 1}), ('progress', {'increment': 2})])
