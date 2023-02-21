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

import sys
import time
import warnings
from copy import deepcopy
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

from deepdiff import DeepHash, Delta

from lightning.app.core.queues import BaseQueue
from lightning.app.storage import Path
from lightning.app.storage.drive import _maybe_create_drive, Drive
from lightning.app.storage.payload import Payload
from lightning.app.utilities.app_helpers import _is_json_serializable, _LightningAppRef, is_overridden
from lightning.app.utilities.app_status import WorkStatus
from lightning.app.utilities.component import _is_flow_context, _sanitize_state
from lightning.app.utilities.enum import (
    CacheCallsKeys,
    make_status,
    WorkFailureReasons,
    WorkStageStatus,
    WorkStopReasons,
)
from lightning.app.utilities.exceptions import LightningWorkException
from lightning.app.utilities.introspection import _is_init_context
from lightning.app.utilities.network import find_free_network_port
from lightning.app.utilities.packaging.build_config import BuildConfig
from lightning.app.utilities.packaging.cloud_compute import (
    _CLOUD_COMPUTE_STORE,
    _CloudComputeStore,
    _maybe_create_cloud_compute,
    CloudCompute,
)
from lightning.app.utilities.proxies import Action, LightningWorkSetAttrProxy, ProxyWorkRun, unwrap, WorkRunExecutor

if TYPE_CHECKING:
    from lightning.app.frontend import Frontend


class LightningWork:
    _INTERNAL_STATE_VARS = (
        # Internal protected variables that are still part of the state (even though they are prefixed with "_")
        "_paths",
        "_host",
        "_port",
        "_url",
        "_restarting",
        "_internal_ip",
    )

    _run_executor_cls: Type[WorkRunExecutor] = WorkRunExecutor
    # TODO: Move to spawn for all Operating System.
    _start_method = "spawn" if sys.platform in ("darwin", "win32") else "fork"

    def __init__(
        self,
        parallel: bool = False,
        cache_calls: bool = True,
        raise_exception: bool = True,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        local_build_config: Optional[BuildConfig] = None,
        cloud_build_config: Optional[BuildConfig] = None,
        cloud_compute: Optional[CloudCompute] = None,
        run_once: Optional[bool] = None,  # TODO: Remove run_once
        start_with_flow: bool = True,
    ):
        """LightningWork, or Work in short, is a building block for long-running jobs.

        The LightningApp runs its :class:`~lightning.app.core.flow.LightningFlow` component
        within an infinite loop and track the ``LightningWork`` status update.

        Use LightningWork for third-party services or for launching heavy jobs such as
        downloading data, training or serving a model.

        Each LightningWork is running in its own independent process. Works are self-isolated from the rest,
        e.g any state changes happening within the work will be reflected within the flow but not the other way around.

        Arguments:
            parallel: Whether to run in parallel mode or not. When False, the flow waits for the work to finish.
            cache_calls: Whether the ``run`` method should cache its input arguments and not run again when provided
                with the same arguments in subsequent calls.
            raise_exception: Whether to re-raise an exception in the flow when raised from within the work run method.
            host: Bind socket to this host
            port: Bind socket to this port. Be default, this is None and should be called within your run method.
            local_build_config: The local BuildConfig isn't used until Lightning supports DockerRuntime.
            cloud_build_config: The cloud BuildConfig enables user to easily configure machine before running this work.
            run_once: Deprecated in favor of cache_calls. This will be removed soon.
            start_with_flow: Whether the work should be started at the same time as the root flow. Only applies to works
                defined in ``__init__``.

        **Learn More About Lightning Work Inner Workings**

        .. raw:: html

            <div class="display-card-container">
                <div class="row">

        .. displayitem::
            :header: The Lightning Work inner workings.
            :description: Learn more Lightning Work.
            :col_css: col-md-4
            :button_link: ../../core_api/lightning_work/index.html
            :height: 180
            :tag: Basic

        .. raw:: html

                </div>
            </div>
            <br />
        """
        from lightning.app.runners.backends.backend import Backend

        if run_once is not None:
            warnings.warn(
                "The `run_once` argument to LightningWork is deprecated in favor of `cache_calls` and will be removed"
                " in the next version. Use `cache_calls` instead."
            )
        self._cache_calls = run_once if run_once is not None else cache_calls
        self._state = {
            "_host",
            "_port",
            "_url",
            "_future_url",
            "_internal_ip",
            "_restarting",
            "_cloud_compute",
            "_display_name",
        }
        self._parallel = parallel
        self._host: str = host
        self._port: Optional[int] = port
        self._url: str = ""
        self._future_url: str = ""  # The cache URL is meant to defer resolving the url values.
        self._internal_ip: str = ""
        # setattr_replacement is used by the multiprocessing runtime to send the latest changes to the main coordinator
        self._setattr_replacement: Optional[Callable[[str, Any], None]] = None
        self._name = ""
        self._display_name = ""
        # The ``self._calls`` is used to track whether the run
        # method with a given set of input arguments has already been called.
        # Example of its usage:
        # {
        #   'latest_call_hash': '167fe2e',
        #   '167fe2e': {
        #       'statuses': [
        #           {'stage': 'pending', 'timestamp': 1659433519.851271},
        #           {'stage': 'running', 'timestamp': 1659433519.956482},
        #           {'stage': 'stopped', 'timestamp': 1659433520.055768}]}
        #        ]
        #    },
        #    ...
        # }
        self._calls = {CacheCallsKeys.LATEST_CALL_HASH: None}
        self._changes = {}
        self._raise_exception = raise_exception
        self._paths = {}
        self._request_queue: Optional[BaseQueue] = None
        self._response_queue: Optional[BaseQueue] = None
        self._restarting = False
        self._start_with_flow = start_with_flow
        self._local_build_config = local_build_config or BuildConfig()
        self._cloud_build_config = cloud_build_config or BuildConfig()
        self._cloud_compute = cloud_compute or CloudCompute()
        # tuple instead of a list so that it cannot be modified without using the setter
        self._lightningignore: Tuple[str, ...] = tuple()
        self._backend: Optional[Backend] = None
        self._check_run_is_implemented()
        self._on_init_end()

    @property
    def url(self) -> str:
        """Returns the current url of the work."""
        return self._url

    @url.setter
    def url(self, url: str) -> None:
        self._url = url

    @property
    def host(self) -> str:
        """Returns the current host of the work."""
        return self._host

    @property
    def port(self) -> int:
        if self._port is None:
            self._port = find_free_network_port()
        return self._port

    @property
    def internal_ip(self) -> str:
        """The internal ip address of this LightningWork, reachable by other Work locally and in the cloud.

        By default, this attribute returns the empty string and the ip address will only be returned once the work runs.
        Locally, the address is 127.0.0.1 and in the cloud it will be determined by the cluster.
        """
        return self._internal_ip

    def _on_init_end(self):
        self._local_build_config.on_work_init(self)
        self._cloud_build_config.on_work_init(self, self._cloud_compute)

    @staticmethod
    def _is_state_attribute(name: str) -> bool:
        """Every public attribute is part of the state by default and all protected (prefixed by '_') or private
        (prefixed by '__') attributes are not.

        Exceptions are listed in the `_INTERNAL_STATE_VARS` class variable.
        """
        return name in LightningWork._INTERNAL_STATE_VARS or not name.startswith("_")

    @property
    def name(self):
        """Returns the name of the LightningWork."""
        return self._name

    @property
    def display_name(self):
        """Returns the display name of the LightningWork in the cloud.

        The display name needs to set before the run method of the work is called.
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: str):
        """Sets the display name of the LightningWork in the cloud."""
        if not self.has_started:
            self._display_name = display_name
        elif self._display_name != display_name:
            raise RuntimeError("The display name can be set only before the work has started.")

    @property
    def cache_calls(self) -> bool:
        """Returns whether the ``run`` method should cache its input arguments and not run again when provided with
        the same arguments in subsequent calls."""
        return self._cache_calls

    @property
    def parallel(self) -> bool:
        """Whether to run in parallel mode or not.

        When parallel is False, the flow waits for the work to finish.
        """
        return self._parallel

    @property
    def local_build_config(self) -> BuildConfig:
        return self._local_build_config

    @local_build_config.setter
    def local_build_config(self, build_config: BuildConfig) -> None:
        self._local_build_config = build_config
        self._local_build_config.on_work_init(self)

    @property
    def cloud_build_config(self) -> BuildConfig:
        """Returns the cloud build config used to prepare the selected cloud hardware."""
        return self._cloud_build_config

    @cloud_build_config.setter
    def cloud_build_config(self, build_config: BuildConfig) -> None:
        self._cloud_build_config = build_config
        self._cloud_build_config.on_work_init(self, cloud_compute=self._cloud_compute)

    @property
    def cloud_compute(self) -> CloudCompute:
        return self._cloud_compute

    @cloud_compute.setter
    def cloud_compute(self, cloud_compute: CloudCompute) -> None:
        """Returns the cloud compute used to select the cloud hardware."""
        # A new ID
        current_id = self._cloud_compute.id
        new_id = cloud_compute.id
        if current_id != new_id:
            compute_store: _CloudComputeStore = _CLOUD_COMPUTE_STORE[current_id]
            compute_store.remove(self.name)
        self._cloud_compute = cloud_compute

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

    @property
    def status(self) -> WorkStatus:
        """Return the current status of the work.

        All statuses are stored in the state.
        """
        call_hash = self._calls[CacheCallsKeys.LATEST_CALL_HASH]
        if call_hash in self._calls:
            statuses = self._calls[call_hash]["statuses"]
            # deltas aren't necessarily coming in the expected order.
            statuses = sorted(statuses, key=lambda x: x["timestamp"])
            latest_status = statuses[-1]
            if latest_status.get("reason") == WorkFailureReasons.TIMEOUT:
                return self._aggregate_status_timeout(statuses)
            return WorkStatus(**latest_status)
        return WorkStatus(stage=WorkStageStatus.NOT_STARTED, timestamp=time.time())

    @property
    def statuses(self) -> List[WorkStatus]:
        """Return all the status of the work."""
        call_hash = self._calls[CacheCallsKeys.LATEST_CALL_HASH]
        if call_hash in self._calls:
            statuses = self._calls[call_hash]["statuses"]
            # deltas aren't necessarily coming in the expected order.
            statuses = sorted(statuses, key=lambda x: x["timestamp"])
            return [WorkStatus(**status) for status in statuses]
        return []

    @property
    def has_started(self) -> bool:
        """Return whether the work has started."""
        return self.status.stage != WorkStageStatus.NOT_STARTED

    @property
    def has_stopped(self) -> bool:
        """Return whether the work has stopped."""
        return self.status.stage == WorkStageStatus.STOPPED

    @property
    def has_succeeded(self) -> bool:
        """Return whether the work has succeeded."""
        return self.status.stage == WorkStageStatus.SUCCEEDED

    @property
    def has_failed(self) -> bool:
        """Return whether the work has failed."""
        return self.status.stage == WorkStageStatus.FAILED

    @property
    def has_timeout(self) -> bool:
        """Return whether the work has time-out."""
        return self.has_failed and self.status.reason == WorkFailureReasons.TIMEOUT

    @property
    def is_running(self) -> bool:
        """Return whether the work is running."""
        return self.status.stage == WorkStageStatus.RUNNING

    @property
    def is_pending(self) -> bool:
        """Return whether the work is pending."""
        return self.status.stage == WorkStageStatus.PENDING

    @property
    def num_timeouts(self) -> int:
        """Return the number of timeout status since the lastest succeeded run."""
        status = self.status
        if status.reason == WorkFailureReasons.TIMEOUT:
            return status.count
        return 0

    @property
    def num_successes(self) -> int:
        """Returns the number of successful runs."""
        # FIXME: Resolve this within  single process runtime.
        run_keys = [key for key in self._calls.keys() if key.startswith("run:")]
        if not run_keys:
            return 0

        has_succeeded_counter = 0
        for run_key in run_keys:
            c = len([s for s in self._calls[run_key]["statuses"] if s["stage"] == WorkStageStatus.SUCCEEDED])
            has_succeeded_counter += c

        return has_succeeded_counter

    def _get_property_if_exists(self, name: str) -> Union[property, None]:
        attr = getattr(self.__class__, name, None)
        return attr if isinstance(attr, property) else None

    def __setattr__(self, name: str, value: Any) -> None:
        property_object = self._get_property_if_exists(name)
        if property_object is not None and property_object.fset is not None:
            property_object.fset(self, value)
        else:
            setattr_fn = getattr(self, "_setattr_replacement", None) or self._default_setattr
            setattr_fn(name, value)

    def _default_setattr(self, name: str, value: Any) -> None:
        from lightning.app.core.flow import LightningFlow

        # Allow the run method to be patched with ProxyWorkRun (done by certain Runtime implementations).
        allowed_to_set_run = name == "run" and (
            isinstance(value, ProxyWorkRun)
            or (unwrap(value) == unwrap(self.run))
            or (isinstance(value, partial) and value.func.__name__ == "_dynamic_run_wrapper")
        )

        is_proxy_setattr = isinstance(value, LightningWorkSetAttrProxy)
        is_init_context = _is_init_context(self)

        if (
            not is_init_context
            and name not in self._state
            and name not in self._paths
            and self._is_state_attribute(name)
            and not allowed_to_set_run
        ):
            raise AttributeError(f"Cannot set attributes that were not defined in __init__: {name}.")

        if isinstance(value, str) and value.startswith("lit://"):
            value = Path(value)

        if self._is_state_attribute(name):
            if isinstance(value, (LightningFlow, LightningWork)):
                raise LightningWorkException(
                    "A ``LightningWork`` isn't allowed to take any children "
                    f"such as ``LightningWork`` or ``LightningFlow``. Found {value}."
                )

            elif isinstance(value, Path):
                value._attach_work(work=self)
                value._attach_queues(self._request_queue, self._response_queue)
                value._name = name
                # In the init context, the full name of the Flow and Work is not known, i.e., we can't serialize
                # the path without losing the information of origin and consumer. Hence, we delay the serialization
                # of the path object until the app is instantiated.
                if not is_init_context:
                    self._paths[name] = value.to_dict()
                self._state.add(name)

            elif isinstance(value, Payload):
                if is_init_context:
                    raise AttributeError("The Payload object should be set only within the run method of the work.")
                value._attach_work(work=self)
                value._name = name
                self._state.add(name)

            elif isinstance(value, Drive):
                value = deepcopy(value)
                value.component_name = self.name
                self._state.add(name)

            elif allowed_to_set_run or is_proxy_setattr:
                # enable overriding the run method (dispatcher)
                pass

            elif _is_json_serializable(value):
                self._state.add(name)

            else:
                raise AttributeError(
                    f"Only JSON-serializable attributes are currently supported"
                    f" (str, int, float, bool, tuple, list, dict etc.) to be part of {self} state. "
                    f"Found the attribute {name} with {value} instead. \n"
                    "HINT: Private attributes defined as follows `self._x = y` won't be shared between components "
                    "and therefore don't need to be JSON-serializable. If you need to include non-JSON serializable "
                    "objects in the state, you can use the `lightning.app.storage.Payload` API."
                )

        super().__setattr__(name, value)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
        except AttributeError as e:
            if str(e).endswith("'_state'"):
                raise AttributeError(f"Did you forget to call super().__init__() in {self}")
            raise e

        if isinstance(attr, ProxyWorkRun):
            return attr

        if callable(attr) and getattr(attr, "__name__", "") == "run":
            # disable while building the class.
            if getattr(self, "_cache_calls", False):
                return self._wrap_run_for_caching(attr)
        return attr

    def __getattr__(self, item):
        if item in self.__dict__.get("_paths", {}) and not _is_init_context(self):
            path = Path.from_dict(self._paths[item])
            path._attach_work(work=self)
            path._attach_queues(self._request_queue, self._response_queue)
            return path
        return self.__getattribute__(item)

    def _call_hash(self, fn, args, kwargs) -> str:
        hash_args = args[1:] if len(args) > 0 and args[0] == self else args
        call_obj = {"args": hash_args, "kwargs": kwargs}
        # Note: Generate a hash as 167fe2e.
        # Seven was selected after checking upon Github default SHA length
        # and to minimize hidden state size.
        return str(DeepHash(call_obj)[call_obj])[:7]

    def _wrap_run_for_caching(self, fn):
        @wraps(fn)
        def new_fn(*args, **kwargs):
            call_hash = self._call_hash(fn, args, kwargs)

            entered = call_hash in self._calls
            returned = entered and "ret" in self._calls[call_hash]

            if returned:
                entry = self._calls[call_hash]
                return entry["ret"]

            self._calls[call_hash] = {}

            result = fn(*args, **kwargs)

            self._calls[call_hash] = {"ret": result}

            return result

        return new_fn

    @property
    def changes(self):
        return self._changes.copy()

    @property
    def state(self):
        """Returns the current state of this LightningWork."""
        return {
            "vars": _sanitize_state({el: getattr(self, el) for el in self._state}),
            # this may have the challenge that ret cannot be pickled, we'll need to handle this
            "calls": self._calls.copy(),
            "changes": {},
        }

    @property
    def state_vars(self):
        return {"vars": _sanitize_state({el: getattr(self, el) for el in self._state})}

    @property
    def state_with_changes(self):
        return {
            "vars": _sanitize_state({el: getattr(self, el) for el in self._state}),
            # this may have the challenge that ret cannot be pickled, we'll need to handle this
            "calls": self._calls.copy(),
            "changes": self.changes,
        }

    def set_state(self, provided_state):
        for k, v in provided_state["vars"].items():
            if isinstance(v, Dict):
                v = _maybe_create_drive(self.name, v)
            if isinstance(v, Dict):
                v = _maybe_create_cloud_compute(v)
            setattr(self, k, v)

        self._changes = provided_state["changes"]

        # Note, this is handled by the flow only.
        if _is_flow_context():
            self._cleanup_calls(provided_state["calls"])

        self._calls = provided_state["calls"]

    @staticmethod
    def _cleanup_calls(calls: Dict[str, Any]):
        # 1: Collect all the in_progress call hashes
        in_progress_call_hash = [k for k in list(calls) if k not in (CacheCallsKeys.LATEST_CALL_HASH)]

        for call_hash in in_progress_call_hash:
            if "statuses" not in calls[call_hash]:
                continue

            # 2: Filter the statuses by timestamp
            statuses = sorted(calls[call_hash]["statuses"], key=lambda x: x["timestamp"])

            # If the latest status is succeeded, then drop everything before.
            if statuses[-1]["stage"] == WorkStageStatus.SUCCEEDED:
                status = statuses[-1]
                status["timestamp"] = int(status["timestamp"])
                calls[call_hash]["statuses"] = [status]
            else:
                # TODO: Some status are being duplicated,
                # this seems related to the StateObserver.
                final_statuses = []
                for status in statuses:
                    if status not in final_statuses:
                        final_statuses.append(status)
                calls[call_hash]["statuses"] = final_statuses

    def start(self):
        """Starts LightingWork component via L.CloudCompute."""
        if self.status.stage == WorkStageStatus.STOPPED:
            raise Exception("A work can be started only once for now.")

        # This enables to start the run method with a phony input and exit.
        self.run(Action(method="start"))

    def run(self, *args, **kwargs):
        """Override to add your own logic.

        Raises:
            LightningPlatformException: If resource exceeds platform quotas or other constraints.
        """

    def on_exception(self, exception: BaseException):
        """Override to customize how to handle exception in the run method."""
        if self._raise_exception:
            raise exception

    def _aggregate_status_timeout(self, statuses: List[Dict]) -> WorkStatus:
        """Method used to return the first request and the total count of timeout after the latest succeeded
        status."""
        succeeded_statuses = [
            status_idx for status_idx, status in enumerate(statuses) if status["stage"] == WorkStageStatus.SUCCEEDED
        ]
        if succeeded_statuses:
            succeed_status_id = succeeded_statuses[-1] + 1
            statuses = statuses[succeed_status_id:]
        timeout_statuses = [status for status in statuses if status.get("reason") == WorkFailureReasons.TIMEOUT]
        assert statuses[0]["stage"] == WorkStageStatus.PENDING
        status = {**timeout_statuses[-1], "timestamp": statuses[0]["timestamp"]}
        return WorkStatus(**status, count=len(timeout_statuses))

    def on_exit(self):
        """Override this hook to add your logic when the work is exiting.

        Note: This hook is not guaranteed to be called when running in the cloud.
        """
        pass

    def stop(self):
        """Stops LightingWork component and shuts down hardware provisioned via L.CloudCompute.

        This can only be called from a ``LightningFlow``.
        """
        if not self._backend:
            raise RuntimeError(f"Only the `LightningFlow` can request this work ({self.name!r}) to stop.")
        if self.status.stage == WorkStageStatus.STOPPED:
            return
        latest_hash = self._calls[CacheCallsKeys.LATEST_CALL_HASH]
        stop_status = make_status(WorkStageStatus.STOPPED, reason=WorkStopReasons.PENDING)
        self._calls[latest_hash]["statuses"].append(stop_status)
        app = _LightningAppRef().get_current()
        self._backend.stop_work(app, self)

    def delete(self):
        """Delete LightingWork component and shuts down hardware provisioned via L.CloudCompute.

        Locally, the work.delete() behaves as work.stop().
        """
        if not self._backend:
            raise Exception(
                "Can't delete the work, it looks like it isn't attached to a LightningFlow. "
                "Make sure to assign the Work to a flow instance."
            )
        app = _LightningAppRef().get_current()
        self._backend.delete_work(app, self)

    def _check_run_is_implemented(self) -> None:
        if not is_overridden("run", instance=self, parent=LightningWork):
            raise TypeError(
                f"The work `{self.__class__.__name__}` is missing the `run()` method. This is required. Implement it"
                " first and then call it in your Flow."
            )

    def _register_cloud_compute(self):
        internal_id = self.cloud_compute.id
        assert internal_id
        if internal_id not in _CLOUD_COMPUTE_STORE:
            _CLOUD_COMPUTE_STORE[internal_id] = _CloudComputeStore(id=internal_id, component_names=[])
        _CLOUD_COMPUTE_STORE[internal_id].add_component_name(self.name)

    def apply_flow_delta(self, delta: Delta):
        """Override to customize how the flow should update the work state."""
        # TODO: Add support for thread safe locking over JSON Serializable objects.
        if any(k not in ["values_changed", "type_changed"] for k in delta.to_dict()):
            raise Exception(
                "A forbidden operation to update the work from the flow was detected."
                f" Found {delta.to_dict()}, only `values_changed` and `type_changes` are currently allowed."
            )

        vars = self.state["vars"] + delta
        for name, value in vars.items():
            property_object = self._get_property_if_exists(name)
            if property_object is not None and property_object.fset is not None:
                property_object.fset(self, value)
            else:
                self._default_setattr(name, value)

    def configure_layout(self) -> Union[None, str, "Frontend"]:
        """Configure the UI of this LightningWork.

        You can either

        1.  Return a single :class:`~lightning.app.frontend.frontend.Frontend` object to serve a user interface
            for this Work.
        2.  Return a string containing a URL to act as the user interface for this Work.
        3.  Return ``None`` to indicate that this Work doesn't currently have a user interface.

        **Example:** Serve a static directory (with at least a file index.html inside).

        .. code-block:: python

            from lightning.app.frontend import StaticWebFrontend


            class Work(LightningWork):
                def configure_layout(self):
                    return StaticWebFrontend("path/to/folder/to/serve")

        **Example:** Arrange the UI of my children in tabs (default UI by Lightning).

        .. code-block:: python

            class Work(LightningWork):
                def configure_layout(self):
                    return [
                        dict(name="First Tab", content=self.child0),
                        dict(name="Second Tab", content=self.child1),
                        dict(name="Lightning", content="https://lightning.ai"),
                    ]

        If you don't implement ``configure_layout``, Lightning will use ``self.url``.

        Note:
            This hook gets called at the time of app creation and then again as part of the loop. If desired, a
            returned URL can depend on the state. This is not the case if the work returns a
            :class:`~lightning.app.frontend.frontend.Frontend`. These need to be provided at the time of app creation
            in order for the runtime to start the server.
        """
