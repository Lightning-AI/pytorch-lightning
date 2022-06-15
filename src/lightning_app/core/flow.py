import inspect
from copy import deepcopy
from datetime import datetime
from types import FrameType
from typing import Any, cast, Dict, Generator, Iterable, List, Optional, Tuple, Union

from deepdiff import DeepHash

from lightning_app.core.work import LightningWork
from lightning_app.frontend import Frontend
from lightning_app.storage import Path
from lightning_app.storage.drive import _maybe_create_drive, Drive
from lightning_app.utilities.app_helpers import _is_json_serializable, _LightningAppRef, _set_child_name
from lightning_app.utilities.component import _sanitize_state
from lightning_app.utilities.exceptions import ExitAppException
from lightning_app.utilities.introspection import _is_init_context, _is_run_context


class LightningFlow:

    _INTERNAL_STATE_VARS = {
        # Internal protected variables that are still part of the state (even though they are prefixed with "_")
        "_paths",
        "_layout",
    }

    def __init__(self):
        """The LightningFlow is a building block to coordinate and manage long running-tasks contained within
        :class:`~lightning_app.core.work.LightningWork` or nested LightningFlow.

        At a minimum, a LightningFlow is characterized by:

        * A set of state variables.
        * Long-running jobs (:class:`~lightning_app.core.work.LightningWork`).
        * Its children ``LightningFlow`` or ``LightningWork`` with their state variables.

        **State variables**

        The LightningFlow are special classes whose attributes require to be
        json-serializable (e.g., int, float, bool, list, dict, ...).

        They also may not reach into global variables unless they are constant.

        .. note ::
            The limitation to primitive types will be lifted in time for
            certain aggregate types, and will be made extensible so that component
            developers will be able to add custom state-compatible types.

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

        The actions themselves are self-contained within :class:`~lightning_app.core.work.LightningWork`.
        The :class:`~lightning_app.core.work.LightningWork` are typically used for long-running jobs,
        like downloading a dataset, performing a query, starting a computationally heavy script.
        While one may access any state variable in a LightningWork from a LightningFlow, one may not
        directly call methods of other components from within a LightningWork as LightningWork can't have any children.
        This limitation allows applications to be distributed at scale.

        **Component hierarchy and App**

        Given the above characteristics, a root LightningFlow, potentially containing
        children components, can be passed to an App object and its execution
        can be distributed (each LightningWork will be run within its own process
        or different arrangements).

        .. doctest::

            >>> from lightning_app import LightningFlow
            >>> class RootFlow(LightningFlow):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.counter = 0
            ...     def run(self):
            ...         self.counter += 1
            >>> flow = RootFlow()
            >>> flow.run()
            >>> assert flow.counter == 1
            >>> assert flow.state["vars"]["counter"] == 1
        """
        from lightning_app.runners.backends.backend import Backend

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

    @property
    def name(self):
        """Return the current LightningFlow name."""
        return self._name or "root"

    def __setattr__(self, name, value):
        from lightning_app.structures import Dict, List

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

            if isinstance(value, LightningFlow):
                self._flows.add(name)
                _set_child_name(self, value, name)
                if name in self._state:
                    self._state.remove(name)
                # Attach the backend to the flow and its children work.
                if self._backend:
                    LightningFlow._attach_backend(value, self._backend)

            elif isinstance(value, LightningWork):
                self._works.add(name)
                _set_child_name(self, value, name)
                if name in self._state:
                    self._state.remove(name)
                if self._backend:
                    self._backend._wrap_run_method(_LightningAppRef().get_current(), value)

            elif isinstance(value, (Dict, List)):
                value._backend = self._backend
                self._structures.add(name)
                _set_child_name(self, value, name)
                if self._backend:
                    for flow in value.flows:
                        LightningFlow._attach_backend(flow, self._backend)
                    for work in value.works:
                        self._backend._wrap_run_method(_LightningAppRef().get_current(), work)

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

        for child_flow in flow.flows.values():
            LightningFlow._attach_backend(child_flow, backend)

        for struct_name in flow._structures:
            structure = getattr(flow, struct_name)
            for flow in structure.flows:
                LightningFlow._attach_backend(flow, backend)
            for work in structure.works:
                backend._wrap_run_method(_LightningAppRef().get_current(), work)

        for name in flow._structures:
            getattr(flow, name)._backend = backend

        for work in flow.works(recurse=False):
            backend._wrap_run_method(_LightningAppRef().get_current(), work)

    def __getattr__(self, item):
        if item in self.__dict__.get("_paths", {}):
            return Path.from_dict(self._paths[item])
        return self.__getattribute__(item)

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
    def flows(self):
        """Return its children LightningFlow."""
        return {el: getattr(self, el) for el in sorted(self._flows)}

    def works(self, recurse: bool = True) -> List[LightningWork]:
        """Return its :class:`~lightning_app.core.work.LightningWork`."""
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
        """Return its :class:`~lightning_app.core.work.LightningWork` with their names."""
        named_works = [(el, getattr(self, el)) for el in sorted(self._works)]
        if not recurse:
            return named_works
        for child_name in sorted(self._flows):
            for w in getattr(self, child_name).works(recurse=recurse):
                named_works.append(w)
        for struct_name in sorted(self._structures):
            for w in getattr(self, struct_name).works:
                named_works.append((w.name, w))
        return named_works

    def get_all_children_(self, children):
        sorted_children = sorted(self._flows)
        children.extend([getattr(self, el) for el in sorted_children])
        for child in sorted_children:
            getattr(self, child).get_all_children_(children)
        return children

    def get_all_children(self):
        children = []
        self.get_all_children_(children)
        return children

    def set_state(self, provided_state: Dict) -> None:
        """Method to set the state to this LightningFlow, its children and
        :class:`~lightning_app.core.work.LightningWork`."""
        for k, v in provided_state["vars"].items():
            if isinstance(v, Dict):
                v = _maybe_create_drive(self.name, v)
            setattr(self, k, v)
        self._changes = provided_state["changes"]
        self._calls.update(provided_state["calls"])
        for child, state in provided_state["flows"].items():
            getattr(self, child).set_state(state)
        for work, state in provided_state["works"].items():
            getattr(self, work).set_state(state)
        for structure, state in provided_state["structures"].items():
            getattr(self, structure).set_state(state)

    def _exit(self, end_msg: str = "") -> None:
        """Private method used to exit the application."""
        if end_msg:
            print(end_msg)
        raise ExitAppException

    @staticmethod
    def _is_state_attribute(name: str) -> bool:
        """Every public attribute is part of the state by default and all protected (prefixed by '_') or private
        (prefixed by '__') attributes are not.

        Exceptions are listed in the `_INTERNAL_STATE_VARS` class variable.
        """
        return name in LightningFlow._INTERNAL_STATE_VARS or not name.startswith("_")

    def run(self, *args, **kwargs) -> None:
        pass

    def schedule(
        self, cron_pattern: str, start_time: Optional[datetime] = None, user_key: Optional[str] = None
    ) -> bool:
        """The schedule method is used to run a part of the flow logic on timely manner.

        .. code-block:: python

            from lightning_app import LightningFlow

            class Flow(LightningFlow):

                def run(self):
                    if self.schedule("hourly"):
                        # run some code once every hour.

        Arguments:
            cron_pattern: The cron pattern to provide. Learn more at https://crontab.guru/.
            start_time: The start time of the cron job.
            user_key: Optional key used to improve the caching mechanism.

        A best practice is to avoid running a dynamic flow or work under the self.schedule method.
        Instead, instantiate them within the condition, but run them outside.

         .. code-block:: python

            from lightning_app import LightningFlow
            from lightning_app.structures import List

            class SchedulerDAG(LightningFlow):

                def __init__(self):
                    super().__init__()
                    self.dags = List()

                def run(self):
                    if self.schedule("@hourly"):
                        self.dags.append(DAG(...))

                    for dag in self.dags:
                        payload = dag.run()
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

        1.  Return a single :class:`~lightning_app.frontend.frontend.Frontend` object to serve a user interface
            for this Flow.
        2.  Return a single dictionary to expose the UI of a child flow.
        3.  Return a list of dictionaries to arrange the children of this flow in one or multiple tabs.

        **Example:** Serve a static directory (with at least a file index.html inside).

        .. code-block:: python

            from lightning_app.frontend import StaticWebFrontend

            class Flow(LightningFlow):
                ...
                def configure_layout(self):
                    return StaticWebFrontend("path/to/folder/to/serve")

        **Example:** Serve a streamlit UI (needs the streamlit package to be installed).

        .. code-block:: python

            from lightning_app.frontend import StaticWebFrontend

            class Flow(LightningFlow):
                ...
                def configure_layout(self):
                    return StreamlitFrontend(render_fn=my_streamlit_ui)

            def my_streamlit_ui(state):
                # add your streamlit code here!

        **Example:** Arrange the UI of my children in tabs (default UI by Lightning).

        .. code-block:: python

            class Flow(LightningFlow):
                ...
                def configure_layout(self):
                    return [
                        dict(name="First Tab", content=self.child0),
                        dict(name="Second Tab", content=self.child1),
                        ...
                        # You can include direct URLs too
                        dict(name="Lightning", content="https://lightning.ai"),
                    ]

        If you don't implement ``configure_layout``, Lightning will collect all children and display their UI in a tab
        (if they have their own ``configure_layout`` implemented).

        Note:
            This hook gets called at the time of app creation and then again as part of the loop. If desired, the
            returned layout configuration can depend on the state. The only exception are the flows that return a
            :class:`~lightning_app.frontend.frontend.Frontend`. These need to be provided at the time of app creation
            in order for the runtime to start the server.
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
