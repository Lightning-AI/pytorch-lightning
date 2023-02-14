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

import inspect
import warnings
from copy import deepcopy
from datetime import datetime
from types import FrameType
from typing import Any, cast, Dict, Generator, Iterable, List, Optional, Tuple, Union

from deepdiff import DeepHash

from lightning.app.core.work import LightningWork
from lightning.app.frontend import Frontend
from lightning.app.storage import Path
from lightning.app.storage.drive import _maybe_create_drive, Drive
from lightning.app.utilities.app_helpers import _is_json_serializable, _LightningAppRef, _set_child_name, is_overridden
from lightning.app.utilities.component import _sanitize_state
from lightning.app.utilities.exceptions import ExitAppException
from lightning.app.utilities.introspection import _is_init_context, _is_run_context
from lightning.app.utilities.packaging.cloud_compute import _maybe_create_cloud_compute, CloudCompute


class LightningFlow:
    _INTERNAL_STATE_VARS = {
        # Internal protected variables that are still part of the state (even though they are prefixed with "_")
        "_paths",
        "_layout",
    }

    def __init__(self):
        """The LightningFlow is used by the :class:`~lightning.app.core.app.LightningApp` to coordinate and manage
        long- running jobs contained, the :class:`~lightning.app.core.work.LightningWork`.

        A LightningFlow is characterized by:

        * A set of state variables.
        * Long-running jobs (:class:`~lightning.app.core.work.LightningWork`).
        * Its children ``LightningFlow`` or ``LightningWork`` with their state variables.

        **State variables**

        The LightningFlow are special classes whose attributes require to be
        json-serializable (e.g., int, float, bool, list, dict, ...).

        They also may not reach into global variables unless they are constant.

        The attributes need to be all defined in `__init__` method,
        and eventually assigned to different values throughout the lifetime of the object.
        However, defining new attributes outside of `__init__` is not allowed.

        Attributes taken together represent the state of the component.
        Components are capable of retrieving their state and that of their
        children recursively at any time. They are also capable of setting
        an externally provided state recursively to its children.

        **Execution model and work**

        The entry point for execution is the ``run`` method at the root component.
        The ``run`` method of the root component may call the ``run`` method of its children, and the children
        may call the ``run`` methods of their children and so on.

        The ``run`` method of the root component is called repeatedly in a while loop forever until the app gets
        terminated. In this programming model (reminiscent of React, Vue or Streamlit from the JavaScript world),
        the values of the state variables, or their changes, are translated into actions throughout the component
        hierarchy. This means the flow of execution will only be affected by state changes in a component or one of
        its children, and otherwise remain idempotent.

        The actions themselves are self-contained within :class:`~lightning.app.core.work.LightningWork`.
        The :class:`~lightning.app.core.work.LightningWork` are typically used for long-running jobs,
        like downloading a dataset, performing a query, starting a computationally heavy script.
        While one may access any state variable in a LightningWork from a LightningFlow, one may not
        directly call methods of other components from within a LightningWork as LightningWork can't have any children.
        This limitation allows applications to be distributed at scale.

        **Component hierarchy and App**

        Given the above characteristics, a root LightningFlow, potentially containing
        children components, can be passed to an App object and its execution
        can be distributed (each LightningWork will be run within its own process
        or different arrangements).

        Example:

            >>> from lightning.app import LightningFlow
            >>> class RootFlow(LightningFlow):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.counter = 0
            ...     def run(self):
            ...         self.counter += 1
            ...
            >>> flow = RootFlow()
            >>> flow.run()
            >>> assert flow.counter == 1
            >>> assert flow.state["vars"]["counter"] == 1
        """
        from lightning.app.runners.backends.backend import Backend

        self._state = set()
        self._name = ""
        self._flows = set()
        self._works = set()
        self._structures = set()
        self._calls = {}
        self._changes = {}
        self._layout: Union[List[Dict], Dict] = {}
        self._paths = {}
        self._backend: Optional[Backend] = None
        # tuple instead of a list so that it cannot be modified without using the setter
        self._lightningignore: Tuple[str, ...] = tuple()

    @property
    def name(self):
        """Return the current LightningFlow name."""
        return self._name or "root"

    def __setattr__(self, name: str, value: Any) -> None:
        attr = getattr(self.__class__, name, None)
        if isinstance(attr, property) and attr.fset is not None:
            return attr.fset(self, value)

        from lightning.app.structures import Dict, List

        if (
            not _is_init_context(self)
            and name not in self._state
            and name not in self._paths
            and (
                not isinstance(value, (LightningWork, LightningFlow))
                or (isinstance(value, (LightningWork, LightningFlow)) and not _is_run_context(self))
            )
            and name not in self._works.union(self._flows)
            and self._is_state_attribute(name)
        ):
            raise AttributeError(f"Cannot set attributes that were not defined in __init__: {name}")

        if isinstance(value, str) and value.startswith("lit://"):
            value = Path(value)

        if self._is_state_attribute(name):

            if hasattr(self, name):
                if name in self._flows and value != getattr(self, name):
                    raise AttributeError(f"Cannot set attributes as the flow can't be changed once defined: {name}")

                if name in self._works and value != getattr(self, name):
                    raise AttributeError(f"Cannot set attributes as the work can't be changed once defined: {name}")

            if isinstance(value, (list, dict)) and value:
                _type = (LightningFlow, LightningWork, List, Dict)
                if isinstance(value, list) and all(isinstance(va, _type) for va in value):
                    value = List(*value)

                if isinstance(value, dict) and all(isinstance(va, _type) for va in value.values()):
                    value = Dict(**value)

            if isinstance(value, LightningFlow):
                self._flows.add(name)
                _set_child_name(self, value, name)
                if name in self._state:
                    self._state.remove(name)
                # Attach the backend to the flow and its children work.
                if self._backend:
                    LightningFlow._attach_backend(value, self._backend)
                for work in value.works():
                    work._register_cloud_compute()

            elif isinstance(value, LightningWork):
                self._works.add(name)
                _set_child_name(self, value, name)
                if name in self._state:
                    self._state.remove(name)
                if self._backend:
                    self._backend._wrap_run_method(_LightningAppRef().get_current(), value)
                value._register_cloud_compute()

            elif isinstance(value, (Dict, List)):
                self._structures.add(name)
                _set_child_name(self, value, name)

                _backend = getattr(self, "backend", None)
                if _backend is not None:
                    value._backend = _backend

                for flow in value.flows:
                    if _backend is not None:
                        LightningFlow._attach_backend(flow, _backend)

                for work in value.works:
                    work._register_cloud_compute()
                    if _backend is not None:
                        _backend._wrap_run_method(_LightningAppRef().get_current(), work)

            elif isinstance(value, Path):
                # In the init context, the full name of the Flow and Work is not known, i.e., we can't serialize
                # the path without losing the information of origin and consumer. Hence, we delay the serialization
                # of the path object until the app is instantiated.
                if not _is_init_context(self):
                    self._paths[name] = value.to_dict()
                self._state.add(name)

            elif isinstance(value, Drive):
                value = deepcopy(value)
                value.component_name = self.name
                self._state.add(name)

            elif isinstance(value, CloudCompute):
                self._state.add(name)

            elif _is_json_serializable(value):
                self._state.add(name)

                if not isinstance(value, Path) and hasattr(self, "_paths") and name in self._paths:
                    # The attribute changed type from Path to another
                    self._paths.pop(name)

            else:
                raise AttributeError(
                    f"Only JSON-serializable attributes are currently supported"
                    f" (str, int, float, bool, tuple, list, dict etc.) to be part of {self} state. "
                    f"Found the attribute {name} with {value} instead. \n"
                    "HINT: Private attributes defined as follows `self._x = y` won't be shared between components "
                    "and therefore don't need to be JSON-serializable."
                )

        super().__setattr__(name, value)

    @staticmethod
    def _attach_backend(flow: "LightningFlow", backend):
        """Attach the backend to all flows and its children."""
        flow._backend = backend

        for name in flow._structures:
            getattr(flow, name)._backend = backend

        for child_flow in flow.flows.values():
            child_flow._backend = backend
            for name in child_flow._structures:
                getattr(child_flow, name)._backend = backend

        app = _LightningAppRef().get_current()

        for child_work in flow.works():
            child_work._backend = backend
            backend._wrap_run_method(app, child_work)

    def __getattr__(self, item):
        if item in self.__dict__.get("_paths", {}):
            return Path.from_dict(self._paths[item])
        return self.__getattribute__(item)

    @property
    def ready(self) -> bool:
        """Override to customize when your App should be ready."""
        flows = self.flows
        return all(flow.ready for flow in flows.values()) if flows else True

    @property
    def changes(self):
        return self._changes.copy()

    @property
    def state(self):
        """Returns the current flow state along its children."""
        children_state = {child: getattr(self, child).state for child in self._flows}
        works_state = {work: getattr(self, work).state for work in self._works}
        return {
            "vars": _sanitize_state({el: getattr(self, el) for el in self._state}),
            # this may have the challenge that ret cannot be pickled, we'll need to handle this
            "calls": self._calls.copy(),
            "flows": children_state,
            "works": works_state,
            "structures": {child: getattr(self, child).state for child in self._structures},
            "changes": {},
        }

    @property
    def state_vars(self):
        children_state = {child: getattr(self, child).state_vars for child in self._flows}
        works_state = {work: getattr(self, work).state_vars for work in self._works}
        return {
            "vars": _sanitize_state({el: getattr(self, el) for el in self._state}),
            "flows": children_state,
            "works": works_state,
            "structures": {child: getattr(self, child).state_vars for child in self._structures},
        }

    @property
    def state_with_changes(self):
        children_state = {child: getattr(self, child).state_with_changes for child in self._flows}
        works_state = {work: getattr(self, work).state_with_changes for work in self._works}
        return {
            "vars": _sanitize_state({el: getattr(self, el) for el in self._state}),
            # this may have the challenge that ret cannot be pickled, we'll need to handle this
            "calls": self._calls.copy(),
            "flows": children_state,
            "works": works_state,
            "structures": {child: getattr(self, child).state_with_changes for child in self._structures},
            "changes": self.changes,
        }

    @property
    def flows(self) -> Dict[str, "LightningFlow"]:
        """Return its children LightningFlow."""
        flows = {}
        for el in sorted(self._flows):
            flow = getattr(self, el)
            flows[flow.name] = flow
            flows.update(flow.flows)
        for struct_name in sorted(self._structures):
            flows.update(getattr(self, struct_name).flows)
        return flows

    @property
    def lightningignore(self) -> Tuple[str, ...]:
        """Programmatic equivalent of the ``.lightningignore`` file."""
        return self._lightningignore

    @lightningignore.setter
    def lightningignore(self, lightningignore: Tuple[str, ...]) -> None:
        if self._backend is not None:
            raise RuntimeError(
                f"Your app has been already dispatched, so modifying the `{self.name}.lightningignore` does not have an"
                " effect"
            )
        self._lightningignore = lightningignore

    def works(self, recurse: bool = True) -> List[LightningWork]:
        """Return its :class:`~lightning.app.core.work.LightningWork`."""
        works = [getattr(self, el) for el in sorted(self._works)]
        if not recurse:
            return works
        for child_name in sorted(self._flows):
            for w in getattr(self, child_name).works(recurse=recurse):
                works.append(w)
        for struct_name in sorted(self._structures):
            for w in getattr(self, struct_name).works:
                works.append(w)
        return works

    def named_works(self, recurse: bool = True) -> List[Tuple[str, LightningWork]]:
        """Return its :class:`~lightning.app.core.work.LightningWork` with their names."""
        return [(w.name, w) for w in self.works(recurse=recurse)]

    def set_state(self, provided_state: Dict, recurse: bool = True) -> None:
        """Method to set the state to this LightningFlow, its children and
        :class:`~lightning.app.core.work.LightningWork`.

        Arguments:
            provided_state: The state to be reloaded
            recurse: Whether to apply the state down children.
        """
        for k, v in provided_state["vars"].items():
            if isinstance(v, Dict):
                v = _maybe_create_drive(self.name, v)
            if isinstance(v, Dict):
                v = _maybe_create_cloud_compute(v)
            setattr(self, k, v)
        self._changes = provided_state["changes"]
        self._calls.update(provided_state["calls"])

        if not recurse:
            return

        for child, state in provided_state["flows"].items():
            getattr(self, child).set_state(state)
        for work, state in provided_state["works"].items():
            getattr(self, work).set_state(state)
        for structure, state in provided_state["structures"].items():
            getattr(self, structure).set_state(state)

    def stop(self, end_msg: str = "") -> None:
        """Method used to exit the application."""
        if end_msg:
            print(end_msg)
        raise ExitAppException

    def _exit(self, end_msg: str = "") -> None:
        """Used to exit the application.

        Private method.

        .. deprecated:: 1.9.0
            This function is deprecated and will be removed in 2.0.0. Use :meth:`stop` instead.
        """
        warnings.warn(
            DeprecationWarning(
                "This function is deprecated and will be removed in 2.0.0. Use `LightningFlow.stop` instead."
            )
        )

        return self.stop(end_msg=end_msg)

    @staticmethod
    def _is_state_attribute(name: str) -> bool:
        """Every public attribute is part of the state by default and all protected (prefixed by '_') or private
        (prefixed by '__') attributes are not.

        Exceptions are listed in the `_INTERNAL_STATE_VARS` class variable.
        """
        return name in LightningFlow._INTERNAL_STATE_VARS or not name.startswith("_")

    def run(self, *args, **kwargs) -> None:
        """Override with your own logic."""
        pass

    def schedule(
        self, cron_pattern: str, start_time: Optional[datetime] = None, user_key: Optional[str] = None
    ) -> bool:
        """The schedule method is used to run a part of the flow logic on timely manner.

        .. code-block:: python

            from lightning.app import LightningFlow


            class Flow(LightningFlow):
                def run(self):
                    if self.schedule("hourly"):
                        print("run some code every hour")

        Arguments:
            cron_pattern: The cron pattern to provide. Learn more at https://crontab.guru/.
            start_time: The start time of the cron job.
            user_key: Optional key used to improve the caching mechanism.

        A best practice is to avoid running a dynamic flow or work under the self.schedule method.
        Instead, instantiate them within the condition, but run them outside.

        .. code-block:: python

            from lightning.app import LightningFlow
            from lightning.app.structures import List


            class SchedulerDAG(LightningFlow):
                def __init__(self):
                    super().__init__()
                    self.dags = List()

                def run(self):
                    if self.schedule("hourly"):
                        self.dags.append(DAG(...))

                    for dag in self.dags:
                        payload = dag.run()

        **Learn more about Scheduling**

        .. raw:: html

            <div class="display-card-container">
                <div class="row">

        .. displayitem::
            :header: Schedule your components
            :description: Learn more scheduling.
            :col_css: col-md-4
            :button_link: ../../../glossary/scheduling.html
            :height: 180
            :tag: Basic

        .. displayitem::
            :header: Build your own DAG
            :description: Learn more DAG scheduling with examples.
            :col_css: col-md-4
            :button_link: ../../../examples/app_dag/dag.html
            :height: 180
            :tag: Basic

        .. raw:: html

                </div>
            </div>
            <br />
        """
        if not user_key:
            frame = cast(FrameType, inspect.currentframe()).f_back
            cache_key = f"{cron_pattern}.{frame.f_code.co_filename}.{frame.f_lineno}"
        else:
            cache_key = user_key

        call_hash = f"{self.schedule.__name__}:{DeepHash(cache_key)[cache_key]}"

        if "scheduling" not in self._calls:
            self._calls["scheduling"] = {}

        entered = call_hash in self._calls["scheduling"]

        expr_aliases = {
            "midnight": "@midnight",
            "hourly": "@hourly",
            "daily": "@daily",
            "weekly": "@weekly",
            "monthly": "@monthly",
            "yearly": "@yearly",
            "annually": "@annually",
        }

        if cron_pattern in expr_aliases:
            cron_pattern = expr_aliases[cron_pattern]

        if not entered:
            if not start_time:
                start_time = datetime.now()

            schedule_metadata = {
                "running": False,
                "cron_pattern": cron_pattern,
                "start_time": str(start_time.isoformat()),
                "name": self.name,
            }

            self._calls["scheduling"][call_hash] = schedule_metadata
            app = _LightningAppRef().get_current()
            if app:
                app._register_schedule(call_hash, schedule_metadata)
            return True
        else:
            return self._calls["scheduling"][call_hash]["running"]

    def _enable_schedule(self, call_hash) -> None:
        self._calls["scheduling"][call_hash]["running"] = True

    def _disable_running_schedules(self) -> None:
        if "scheduling" not in self._calls:
            return
        for call_hash in self._calls["scheduling"]:
            self._calls["scheduling"][call_hash]["running"] = False

    def configure_layout(self) -> Union[Dict[str, Any], List[Dict[str, Any]], Frontend]:
        """Configure the UI layout of this LightningFlow.

        You can either

        1.  Return a single :class:`~lightning.app.frontend.frontend.Frontend` object to serve a user interface
            for this Flow.
        2.  Return a single dictionary to expose the UI of a child flow.
        3.  Return a list of dictionaries to arrange the children of this flow in one or multiple tabs.

        **Example:** Serve a static directory (with at least a file index.html inside).

        .. code-block:: python

            from lightning.app.frontend import StaticWebFrontend


            class Flow(LightningFlow):
                ...

                def configure_layout(self):
                    return StaticWebFrontend("path/to/folder/to/serve")

        **Example:** Serve a streamlit UI (needs the streamlit package to be installed).

        .. code-block:: python

            from lightning.app.frontend import StaticWebFrontend


            class Flow(LightningFlow):
                ...

                def configure_layout(self):
                    return StreamlitFrontend(render_fn=my_streamlit_ui)


            def my_streamlit_ui(state):
                # add your streamlit code here!
                import streamlit as st


        **Example:** Arrange the UI of my children in tabs (default UI by Lightning).

        .. code-block:: python

            class Flow(LightningFlow):
                def configure_layout(self):
                    return [
                        dict(name="First Tab", content=self.child0),
                        dict(name="Second Tab", content=self.child1),
                        dict(name="Lightning", content="https://lightning.ai"),
                    ]

        If you don't implement ``configure_layout``, Lightning will collect all children and display their UI in a tab
        (if they have their own ``configure_layout`` implemented).

        Note:
            This hook gets called at the time of app creation and then again as part of the loop. If desired, the
            returned layout configuration can depend on the state. The only exception are the flows that return a
            :class:`~lightning.app.frontend.frontend.Frontend`. These need to be provided at the time of app creation
            in order for the runtime to start the server.

        **Learn more about adding UI**

        .. raw:: html

            <div class="display-card-container">
                <div class="row">

        .. displayitem::
            :header: Add a web user interface (UI)
            :description: Learn more how to integrate several UIs.
            :col_css: col-md-4
            :button_link: ../../../workflows/add_web_ui/index.html
            :height: 180
            :tag: Basic

        .. raw:: html

                </div>
            </div>
            <br />
        """
        return [dict(name=name, content=component) for (name, component) in self.flows.items()]

    def experimental_iterate(self, iterable: Iterable, run_once: bool = True, user_key: str = "") -> Generator:
        """This method should always be used with any kind of iterable to ensure its fault tolerant.

        If you want your iterable to always be consumed from scratch, you shouldn't use this method.

        Arguments:
            iterable: Iterable to iterate over. The iterable shouldn't have side effects or be random.
            run_once: Whether to run the entire iteration only once.
                Otherwise, it would restart from the beginning.
            user_key: Key to be used to track the caching mechanism.
        """
        if not isinstance(iterable, Iterable):
            raise TypeError(f"An iterable should be provided to `self.iterate` method. Found {iterable}")

        # TODO: Find a better way. Investigated using __reduce__, but state change invalidate the cache.
        if not user_key:
            frame = cast(FrameType, inspect.currentframe()).f_back
            cache_key = f"{frame.f_code.co_filename}.{frame.f_code.co_firstlineno}"
        else:
            cache_key = user_key

        call_hash = f"{self.experimental_iterate.__name__}:{DeepHash(cache_key)[cache_key]}"
        entered = call_hash in self._calls
        has_started = entered and self._calls[call_hash]["counter"] > 0
        has_finished = entered and self._calls[call_hash]["has_finished"]

        if has_finished:
            if not run_once:
                self._calls[call_hash].update({"counter": 0, "has_finished": False})
            else:
                return range(0)

        if not has_started:
            self._calls[call_hash] = {
                "name": self.experimental_iterate.__name__,
                "call_hash": call_hash,
                "counter": 0,
                "has_finished": False,
            }

        skip_counter = max(self._calls[call_hash]["counter"], 0)

        for counter, value in enumerate(iterable):
            if skip_counter:
                skip_counter -= 1
                continue
            self._calls[call_hash].update({"counter": counter})
            yield value

        self._calls[call_hash].update({"has_finished": True})

    def configure_commands(self):
        """Configure the commands of this LightningFlow.

        Returns a list of dictionaries mapping a command name to a flow method.

        .. code-block:: python

            class Flow(LightningFlow):
                def __init__(self):
                    super().__init__()
                    self.names = []

                def configure_commands(self):
                    return {"my_command_name": self.my_remote_method}

                def my_remote_method(self, name):
                    self.names.append(name)

        Once the app is running with the following command:

        .. code-block:: bash

            lightning run app app.py

        .. code-block:: bash

            lightning my_command_name --args name=my_own_name
        """
        raise NotImplementedError

    def configure_api(self):
        """Configure the API routes of the LightningFlow.

        Returns a list of HttpMethod such as Post or Get.

        .. code-block:: python

            from lightning.app import LightningFlow
            from lightning.app.api import Post

            from pydantic import BaseModel


            class HandlerModel(BaseModel):
                name: str


            class Flow(L.LightningFlow):
                def __init__(self):
                    super().__init__()
                    self.names = []

                def handler(self, config: HandlerModel) -> None:
                    self.names.append(config.name)

                def configure_api(self):
                    return [Post("/v1/api/request", self.handler)]

        Once the app is running, you can access the Swagger UI of the app
        under the ``/docs`` route.
        """
        raise NotImplementedError

    def state_dict(self):
        """Returns the current flow state but not its children."""
        return {
            "vars": _sanitize_state({el: getattr(self, el) for el in self._state}),
            "calls": self._calls.copy(),
            "changes": {},
            "flows": {},
            "works": {},
            "structures": {},
        }

    def load_state_dict(
        self,
        flow_state: Dict[str, Any],
        children_states: Dict[str, Any],
        strict: bool = True,
    ) -> None:
        """Reloads the state of this flow and its children.

        .. code-block:: python

            import lightning as L


            class Work(L.LightningWork):
                def __init__(self):
                    super().__init__()
                    self.counter = 0

                def run(self):
                    self.counter += 1


            class Flow(L.LightningFlow):
                def run(self):
                    # dynamically create a work.
                    if not getattr(self, "w", None):
                        self.w = WorkReload()

                    self.w.run()

                def load_state_dict(self, flow_state, children_states, strict) -> None:
                    # 1: Re-instantiate the dynamic work
                    self.w = Work()

                    # 2: Make any states modification / migration.
                    ...

                    # 3: Call the parent ``load_state_dict`` to
                    # recursively reload the states.
                    super().load_state_dict(
                        flow_state,
                        children_states,
                        strict,
                    )

        Arguments:
            flow_state: The state of the current flow.
            children_states: The state of the dynamic children of this flow.
            strict: Whether to raise an exception if a dynamic
                children hasn't been re-created.
        """
        self.set_state(flow_state, recurse=False)
        direct_children_states = {k: v for k, v in children_states.items() if "." not in k}
        for child_name, state in direct_children_states.items():
            child = getattr(self, child_name, None)
            if isinstance(child, LightningFlow):
                lower_children_states = {
                    k.replace(child_name + ".", ""): v
                    for k, v in children_states.items()
                    if k.startswith(child_name) and k != child_name
                }
                child.load_state_dict(state, lower_children_states, strict=strict)
            elif isinstance(child, LightningWork):
                child.set_state(state)
            elif strict:
                raise ValueError(f"The component {child_name} wasn't instantiated for the component {self.name}")


class _RootFlow(LightningFlow):
    def __init__(self, work):
        super().__init__()
        self.work = work

    @property
    def ready(self) -> bool:
        ready = getattr(self.work, "ready", None)
        if ready is not None:
            return ready
        return self.work.url != ""

    def run(self):
        if self.work.has_succeeded:
            self.work.stop()
            self.stop()
        self.work.run()

    def configure_layout(self):
        if is_overridden("configure_layout", self.work):
            return [{"name": "Main", "content": self.work}]
        return []
