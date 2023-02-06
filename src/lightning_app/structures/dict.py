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

from lightning_app.utilities.app_helpers import _LightningAppRef, _set_child_name

T = t.TypeVar("T")

if t.TYPE_CHECKING:
    from lightning_app.utilities.types import Component


def _prepare_name(component: "Component") -> str:
    return str(component.name.split(".")[-1])


# TODO: add support and tests for dict operations (insertion, update, etc.)
class Dict(t.Dict[str, T]):
    def __init__(self, **kwargs: T):
        """The Dict Object is used to represents dict collection of
        :class:`~lightning_app.core.work.LightningWork`
        or :class:`~lightning_app.core.flow.LightningFlow`.

        Example:

            >>> from lightning_app import LightningFlow, LightningWork
            >>> from lightning_app.structures import Dict
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
            ...         self.dict = Dict(**{"work_0": CounterWork(), "work_1": CounterWork()})
            ...     def run(self):
            ...         for work_name, work in self.dict.items():
            ...             work.run()
            ...
            >>> flow = RootFlow()
            >>> flow.run()
            >>> assert flow.dict["work_0"].counter == 1

        Arguments:
            items: A sequence of LightningWork or LightningFlow.
        """
        super().__init__(**kwargs)
        from lightning_app.runners.backends import Backend

        self._name: t.Optional[str] = ""
        self._backend: t.Optional[Backend] = None
        for k, v in kwargs.items():
            if "." in k:
                raise Exception(f"The provided name {k} contains . which is forbidden.")
            _set_child_name(self, v, k)

    def __setitem__(self, k, v):
        from lightning_app import LightningFlow, LightningWork

        if not isinstance(k, str):
            raise Exception("The provided key should be an string")

        if isinstance(k, str) and "." in k:
            raise Exception(f"The provided name {k} contains . which is forbidden.")

        _set_child_name(self, v, k)
        if self._backend:
            if isinstance(v, LightningFlow):
                LightningFlow._attach_backend(v, self._backend)
            elif isinstance(v, LightningWork):
                self._backend._wrap_run_method(_LightningAppRef().get_current(), v)
        v._name = f"{self.name}.{k}"
        super().__setitem__(k, v)

    @property
    def works(self):
        from lightning_app import LightningFlow, LightningWork

        works = [item for item in self.values() if isinstance(item, LightningWork)]
        for flow in [item for item in self.values() if isinstance(item, LightningFlow)]:
            for child_work in flow.works(recurse=False):
                works.append(child_work)
        return works

    @property
    def flows(self):
        from lightning_app import LightningFlow
        from lightning_app.structures import Dict, List

        flows = {}
        for item in self.values():
            if isinstance(item, LightningFlow):
                flows[item.name] = item
                for child_flow in item.flows.values():
                    flows[child_flow.name] = child_flow
            if isinstance(item, (Dict, List)):
                for child_flow in item.flows.values():
                    flows[child_flow.name] = child_flow
        return flows

    @property
    def name(self):
        return self._name or "root"

    @property
    def state(self):
        """Returns the state of its flows and works."""
        from lightning_app import LightningFlow, LightningWork

        return {
            "works": {key: item.state for key, item in self.items() if isinstance(item, LightningWork)},
            "flows": {key: item.state for key, item in self.items() if isinstance(item, LightningFlow)},
        }

    @property
    def state_vars(self):
        from lightning_app import LightningFlow, LightningWork

        return {
            "works": {key: item.state_vars for key, item in self.items() if isinstance(item, LightningWork)},
            "flows": {key: item.state_vars for key, item in self.items() if isinstance(item, LightningFlow)},
        }

    @property
    def state_with_changes(self):
        from lightning_app import LightningFlow, LightningWork

        return {
            "works": {key: item.state_with_changes for key, item in self.items() if isinstance(item, LightningWork)},
            "flows": {key: item.state_with_changes for key, item in self.items() if isinstance(item, LightningFlow)},
        }

    def set_state(self, state):
        state_keys = set(list(state["works"].keys()) + list(state["flows"].keys()))
        current_state_keys = set(self.keys())
        if current_state_keys != state_keys:
            key_diff = (current_state_keys - state_keys) | (state_keys - current_state_keys)
            raise Exception(
                f"The provided state doesn't match the `Dict` {self.name}. Found `{key_diff}` un-matching keys"
            )
        for work_key, work_state in state["works"].items():
            self[work_key].set_state(work_state)
        for child_key, child_state in state["flows"].items():
            self[child_key].set_state(child_state)
