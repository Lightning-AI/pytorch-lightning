import abc
import time
import warnings
from copy import deepcopy
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional

from deepdiff import DeepHash

from lightning_app.core.queues import BaseQueue
from lightning_app.storage import Path
from lightning_app.storage.drive import _maybe_create_drive, Drive
from lightning_app.storage.payload import Payload
from lightning_app.utilities.app_helpers import _is_json_serializable, _LightningAppRef
from lightning_app.utilities.component import _sanitize_state
from lightning_app.utilities.enum import make_status, WorkFailureReasons, WorkStageStatus, WorkStatus, WorkStopReasons
from lightning_app.utilities.exceptions import LightningWorkException
from lightning_app.utilities.introspection import _is_init_context
from lightning_app.utilities.network import find_free_network_port
from lightning_app.utilities.packaging.build_config import BuildConfig
from lightning_app.utilities.packaging.cloud_compute import CloudCompute
from lightning_app.utilities.proxies import LightningWorkSetAttrProxy, ProxyWorkRun, unwrap


class LightningWork(abc.ABC):

    _INTERNAL_STATE_VARS = (
        # Internal protected variables that are still part of the state (even though they are prefixed with "_")
        "_paths",
        "_host",
        "_port",
        "_url",
        "_restarting",
        "_internal_ip",
    )

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
    ):
        """LightningWork, or Work in short, is a building block for long-running jobs.

        The LightningApp runs its :class:`~lightning_app.core.flow.LightningFlow` component
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
            port: Bind socket to this port
        """
        from lightning_app.runners.backends.backend import Backend

        if run_once is not None:
            warnings.warn(
                "The `run_once` argument to LightningWork is deprecated in favor of `cache_calls` and will be removed"
                " in the next version. Use `cache_calls` instead."
            )
        self._cache_calls = run_once if run_once is not None else cache_calls
        self._state = {"_host", "_port", "_url", "_future_url", "_internal_ip", "_restarting"}
        self._parallel = parallel
        self._host: str = host
        self._port: Optional[int] = port
        self._url: str = ""
        self._future_url: str = ""  # The cache URL is meant to defer resolving the url values.
        self._internal_ip: str = ""
        # setattr_replacement is used by the multiprocessing runtime to send the latest changes to the main coordinator
        self._setattr_replacement: Optional[Callable[[str, Any], None]] = None
        self._name = ""
        self._calls = {"latest_call_hash": None}
        self._changes = {}
        self._raise_exception = raise_exception
        self._paths = {}
        self._request_queue: Optional[BaseQueue] = None
        self._response_queue: Optional[BaseQueue] = None
        self._restarting = False
        self._local_build_config = local_build_config or BuildConfig()
        self._cloud_build_config = cloud_build_config or BuildConfig()
        self._cloud_compute = cloud_compute or CloudCompute()
        self._backend: Optional[Backend] = None
        self._on_init_end()

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, url: str) -> None:
        self._url = url

    @property
    def host(self) -> str:
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
    def cache_calls(self) -> bool:
        """Returns whether the ``run`` method should cache its input arguments and not run again when provided with the
        same arguments in subsequent calls."""
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
        return self._cloud_build_config

    @cloud_build_config.setter
    def cloud_build_config(self, build_config: BuildConfig) -> None:
        self._cloud_build_config = build_config
        self._cloud_build_config.on_work_init(self, cloud_compute=self._cloud_compute)

    @property
    def cloud_compute(self) -> CloudCompute:
        return self._cloud_compute

    @cloud_compute.setter
    def cloud_compute(self, cloud_compute) -> None:
        self._cloud_compute = cloud_compute

    @property
    def status(self) -> WorkStatus:
        """Return the current status of the work.

        All statuses are stored in the state.
        """
        call_hash = self._calls["latest_call_hash"]
        if call_hash:
            statuses = self._calls[call_hash]["statuses"]
            # deltas aren't necessarily coming in the expected order.
            statuses = sorted(statuses, key=lambda x: x["timestamp"])
            latest_status = statuses[-1]
            if latest_status["reason"] == WorkFailureReasons.TIMEOUT:
                return self._aggregate_status_timeout(statuses)
            return WorkStatus(**latest_status)
        return WorkStatus(stage=WorkStageStatus.NOT_STARTED, timestamp=time.time())

    @property
    def statuses(self) -> List[WorkStatus]:
        """Return all the status of the work."""
        call_hash = self._calls["latest_call_hash"]
        if call_hash:
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
        """Return whether the work has started."""
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

    def __setattr__(self, name: str, value: Any) -> None:
        setattr_fn = getattr(self, "_setattr_replacement", None) or self._default_setattr
        setattr_fn(name, value)

    def _default_setattr(self, name: str, value: Any) -> None:
        from lightning_app.core.flow import LightningFlow

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
                    "objects in the state, you can use the `lightning_app.storage.Payload` API."
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

    def _call_hash(self, fn, args, kwargs):
        hash_args = args[1:] if len(args) > 0 and args[0] == self else args
        call_obj = {"args": hash_args, "kwargs": kwargs}
        return f"{fn.__name__}:{DeepHash(call_obj)[call_obj]}"

    def _wrap_run_for_caching(self, fn):
        @wraps(fn)
        def new_fn(*args, **kwargs):
            call_hash = self._call_hash(fn, args, kwargs)

            entered = call_hash in self._calls
            returned = entered and "ret" in self._calls[call_hash]

            if returned:
                entry = self._calls[call_hash]
                return entry["ret"]

            self._calls[call_hash] = {"name": fn.__name__, "call_hash": call_hash}

            result = fn(*args, **kwargs)

            self._calls[call_hash] = {"name": fn.__name__, "call_hash": call_hash, "ret": result}

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
            setattr(self, k, v)
        self._changes = provided_state["changes"]
        self._calls.update(provided_state["calls"])

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Override to add your own logic."""
        pass

    def on_exception(self, exception: BaseException):
        """Override to customize how to handle exception in the run method."""
        if self._raise_exception:
            raise exception

    def _aggregate_status_timeout(self, statuses: List[Dict]) -> WorkStatus:
        """Method used to return the first request and the total count of timeout after the latest succeeded status."""
        succeeded_statuses = [
            status_idx for status_idx, status in enumerate(statuses) if status["stage"] == WorkStageStatus.SUCCEEDED
        ]
        if succeeded_statuses:
            succeed_status_id = succeeded_statuses[-1] + 1
            statuses = statuses[succeed_status_id:]
        timeout_statuses = [status for status in statuses if status["reason"] == WorkFailureReasons.TIMEOUT]
        assert statuses[0]["stage"] == WorkStageStatus.PENDING
        status = {**timeout_statuses[-1], "timestamp": statuses[0]["timestamp"]}
        return WorkStatus(**status, count=len(timeout_statuses))

    def load_state_dict(self, state):
        # TODO (tchaton) Implement logic for state reloading.
        pass

    def on_exit(self):
        """Override this hook to add your logic when the work is exiting."""
        pass

    def stop(self):
        if not self._backend:
            raise Exception(
                "Can't stop the work, it looks like it isn't attached to a LightningFlow. "
                "Make sure to assign the Work to a flow instance."
            )
        if self.status.stage == WorkStageStatus.STOPPED:
            return
        latest_hash = self._calls["latest_call_hash"]
        self._calls[latest_hash]["statuses"].append(
            make_status(WorkStageStatus.STOPPED, reason=WorkStopReasons.PENDING)
        )
        app = _LightningAppRef().get_current()
        self._backend.stop_work(app, self)
