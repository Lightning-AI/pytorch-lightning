# Copyright The Lightning AI team.
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

import typing as t

from lightning.app.utilities.app_helpers import _LightningAppRef, _set_child_name

T = t.TypeVar("T")

if t.TYPE_CHECKING:
    from lightning.app.utilities.types import Component


def _prepare_name(component: "Component") -> str:
    return str(component.name.split(".")[-1])


# TODO: add support and tests for list operations (concatenation, deletion, insertion, etc.)
class List(t.List[T]):
    def __init__(self, *items: T):
        """The List Object is used to represents list collection of
        :class:`~lightning.app.core.work.LightningWork`
        or :class:`~lightning.app.core.flow.LightningFlow`.

        Example:

            >>> from lightning.app import LightningFlow, LightningWork
            >>> from lightning.app.structures import List
            >>> class CounterWork(LightningWork):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.counter = 0
            ...     def run(self):
            ...         self.counter += 1
            ...
            >>> class RootFlow(LightningFlow):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.list = List(*[CounterWork(), CounterWork()])
            ...     def run(self):
            ...         for work in self.list:
            ...             work.run()
            ...
            >>> flow = RootFlow()
            >>> flow.run()
            >>> assert flow.list[0].counter == 1

        Arguments:
            items: A sequence of LightningWork or LightningFlow.
        """
        super().__init__()
        from lightning.app.runners.backends import Backend

        self._name: t.Optional[str] = ""
        self._last_index = 0
        self._backend: t.Optional[Backend] = None
        for item in items:
            self.append(item)

    def append(self, v):
        from lightning.app import LightningFlow, LightningWork

        _set_child_name(self, v, str(self._last_index))
        if self._backend:
            if isinstance(v, LightningFlow):
                LightningFlow._attach_backend(v, self._backend)
            elif isinstance(v, LightningWork):
                self._backend._wrap_run_method(_LightningAppRef().get_current(), v)
        v._name = f"{self.name}.{self._last_index}"
        self._last_index += 1
        super().append(v)

    @property
    def name(self):
        """Returns the name of this List object."""
        return self._name or "root"

    @property
    def works(self):
        from lightning.app import LightningFlow, LightningWork

        works = [item for item in self if isinstance(item, LightningWork)]
        for flow in [item for item in self if isinstance(item, LightningFlow)]:
            for child_work in flow.works(recurse=False):
                works.append(child_work)
        return works

    @property
    def flows(self):
        from lightning.app import LightningFlow
        from lightning.app.structures import Dict, List

        flows = {}
        for item in self:
            if isinstance(item, LightningFlow):
                flows[item.name] = item
                for child_flow in item.flows.values():
                    flows[child_flow.name] = child_flow
            if isinstance(item, (Dict, List)):
                for child_flow in item.flows.values():
                    flows[child_flow.name] = child_flow
        return flows

    @property
    def state(self):
        """Returns the state of its flows and works."""
        from lightning.app import LightningFlow, LightningWork

        works = [item for item in self if isinstance(item, LightningWork)]
        children = [item for item in self if isinstance(item, LightningFlow)]
        return {
            "works": {_prepare_name(w): w.state for w in works},
            "flows": {_prepare_name(flow): flow.state for flow in children},
        }

    @property
    def state_vars(self):
        from lightning.app import LightningFlow, LightningWork

        works = [item for item in self if isinstance(item, LightningWork)]
        children = [item for item in self if isinstance(item, LightningFlow)]
        return {
            "works": {_prepare_name(w): w.state_vars for w in works},
            "flows": {_prepare_name(flow): flow.state_vars for flow in children},
        }

    @property
    def state_with_changes(self):
        from lightning.app import LightningFlow, LightningWork

        works = [item for item in self if isinstance(item, LightningWork)]
        children = [item for item in self if isinstance(item, LightningFlow)]
        return {
            "works": {str(_prepare_name(w)): w.state_with_changes for w in works},
            "flows": {_prepare_name(flow): flow.state_with_changes for flow in children},
        }

    def set_state(self, state):
        """Method to set the state of the list and its children."""
        from lightning.app import LightningFlow, LightningWork

        works = [item for item in self if isinstance(item, LightningWork)]
        children = [item for item in self if isinstance(item, LightningFlow)]

        current_state_keys = {_prepare_name(w) for w in self}
        state_keys = set(list(state["works"].keys()) + list(state["flows"].keys()))

        if current_state_keys != state_keys:
            key_diff = (current_state_keys - state_keys) | (state_keys - current_state_keys)
            raise Exception(
                f"The provided state doesn't match the `List` {self.name}. Found `{key_diff}` un-matching keys"
            )

        for work_key, work_state in state["works"].items():
            for work in works:
                if _prepare_name(work) == work_key:
                    work.set_state(work_state)
        for child_key, child_state in state["flows"].items():
            for child in children:
                if _prepare_name(child) == child_key:
                    child.set_state(child_state)

    def __len__(self):
        """Returns the number of elements within this List."""
        return len([v for v in self])
