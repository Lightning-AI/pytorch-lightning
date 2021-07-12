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
from pytorch_lightning.trainer.trainer import Trainer


def _collect_loop_progress(loop: Loop) -> Dict[str, Any]:
    """Return the progress for the current loop and its children."""
    progress = {}
    for k, v in loop.__dict__.items():
        if isinstance(v, BaseProgress):
            progress[k] = v
        elif isinstance(v, Loop):
            progress[k] = _collect_loop_progress(v)
    return progress


def test_loop_restore():

    class CustomExpection(Exception):
        pass

    class Simple(Loop):

        def __init__(self, dataset: Iterator):
            super().__init__()
            self.dataset = dataset

        @property
        def skip(self) -> bool:
            return False

        @property
        def done(self) -> bool:
            return self.iteration_count > len(self.dataset)

        def reset(self) -> None:
            self.iter_dataset = iter(self.dataset)

            if self.restarting:
                for _ in range(self.iteration_count):
                    next(self.iter_dataset)
                self.iteration_count += 1
                self.restarting = False
            else:
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

    trainer = Trainer()

    data = range(10)
    loop = Simple(data)
    loop.trainer = trainer
    try:
        loop.run()
        state_dict = {}
    except CustomExpection:
        state_dict = loop.state_dict()

    loop = Simple(data)
    loop.trainer = trainer
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
            loop = getattr(self, "loop_child", None)
            if not loop:
                return
            loop.run()
            self.progress.increment += 1

        @property
        def skip(self) -> bool:
            return False

        @property
        def done(self) -> bool:
            return self.iteration_count > 0

        def reset(self) -> None:
            self.restarting = False

        def on_save_checkpoint(self) -> Dict:
            return {"a": self.a}

        def on_load_checkpoint(self, state_dict: Dict) -> None:
            self.a = state_dict["a"]

    grand_loop_parent = Simple(0)
    loop_parent = Simple(1)
    loop_child = Simple(2)

    loop_parent.loop_child = loop_child

    assert not loop_parent.skip

    state_dict = loop_parent.state_dict()

    loop_progress = _collect_loop_progress(loop_parent)
    assert loop_progress["progress"] == loop_parent.progress
    assert loop_progress["loop_child"]["progress"] == loop_child.progress

    loop_progress = _collect_loop_progress(loop_child)
    assert loop_progress["progress"] == loop_child.progress

    loop_parent.trainer = Trainer()
    assert loop_child.trainer == loop_parent.trainer

    assert state_dict == OrderedDict([('state_dict', {
        'a': 1
    }), ('progress', {
        'increment': 0
    }), ('loop_child.state_dict', {
        'a': 2
    }), ('loop_child.progress', {
        'increment': 0
    })])

    loop_parent.progress

    state_dict["loop_child.state_dict"]["a"] = 3

    loop_parent.load_state_dict(state_dict)
    assert loop_parent.restarting

    loop_parent.run()

    loop_parent_copy = deepcopy(loop_parent)
    assert loop_parent_copy.state_dict() == loop_parent.state_dict()

    assert loop_parent_copy.on_save_checkpoint() == {'a': 1}
    assert loop_parent_copy.loop_child.on_save_checkpoint() == {'a': 3}

    assert not loop_parent.restarting

    state_dict = loop_parent.state_dict()
    assert state_dict == OrderedDict([('state_dict', {
        'a': 1
    }), ('progress', {
        'increment': 1
    }), ('loop_child.state_dict', {
        'a': 3
    }), ('loop_child.progress', {
        'increment': 0
    })])

    loop_parent = Simple(1)
    loop_child = Simple(2)
    loop_parent.loop_child = loop_child
    loop_parent.load_state_dict(state_dict)
    assert loop_parent.progress.increment == 1
    assert loop_parent.loop_child.progress.increment == 0

    del loop_parent.loop_child
    state_dict = loop_parent.state_dict()
    assert state_dict == OrderedDict([('state_dict', {'a': 1}), ('progress', {'increment': 1})])

    grand_loop_parent = Simple(0)
    loop_parent = Simple(1)
    loop_child = Simple(2)
    grand_loop_parent.loop_child = loop_parent
    loop_parent.loop_child = loop_child

    grand_loop_parent.trainer = Trainer()
    assert loop_child.trainer is not None
