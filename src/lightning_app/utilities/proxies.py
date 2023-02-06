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

import os
import pathlib
import queue
import signal
import sys
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from threading import Event, Thread
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, TYPE_CHECKING, Union

from deepdiff import DeepDiff, Delta
from lightning_utilities.core.apply_func import apply_to_collection

from lightning_app.core import constants
from lightning_app.core.queues import MultiProcessQueue
from lightning_app.storage import Path
from lightning_app.storage.copier import _Copier, _copy_files
from lightning_app.storage.drive import _maybe_create_drive, Drive
from lightning_app.storage.path import _path_to_work_artifact
from lightning_app.storage.payload import Payload
from lightning_app.utilities.app_helpers import affiliation
from lightning_app.utilities.component import _set_work_context
from lightning_app.utilities.enum import (
    CacheCallsKeys,
    make_status,
    WorkFailureReasons,
    WorkStageStatus,
    WorkStopReasons,
)
from lightning_app.utilities.exceptions import CacheMissException, LightningSigtermStateException

if TYPE_CHECKING:
    from lightning_app import LightningWork
    from lightning_app.core.queues import BaseQueue

from lightning_app.utilities.app_helpers import Logger

logger = Logger(__name__)
_state_observer_lock = threading.Lock()


@dataclass
class Action:
    method: str = "run"
    args: Tuple = field(default_factory=lambda: ())
    kwargs: Dict = field(default_factory=lambda: {})


def unwrap(fn):
    if isinstance(fn, partial):
        fn = fn.keywords["work_run"]
    if isinstance(fn, ProxyWorkRun):
        fn = fn.work_run
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _send_data_to_caller_queue(
    proxy, work: "LightningWork", caller_queue: "BaseQueue", data: Dict, call_hash: str
) -> Dict:

    proxy.has_sent = True

    if work._calls[CacheCallsKeys.LATEST_CALL_HASH] is None:
        work._calls[CacheCallsKeys.LATEST_CALL_HASH] = call_hash

    if call_hash not in work._calls:
        work._calls[call_hash] = {"statuses": []}
    else:
        # remove ret when relaunching the work.
        work._calls[call_hash].pop("ret", None)

    work._calls[call_hash]["statuses"].append(make_status(WorkStageStatus.PENDING))

    work_state = work.state

    # There is no need to send all call hashes to the work.
    calls = deepcopy(work_state["calls"])
    work_state["calls"] = {
        k: v for k, v in work_state["calls"].items() if k in (call_hash, CacheCallsKeys.LATEST_CALL_HASH)
    }

    data.update({"state": work_state})
    logger.debug(f"Sending to {work.name}: {data}")
    caller_queue.put(deepcopy(data))

    # Reset the calls entry.
    work_state["calls"] = calls
    work._restarting = False
    return work_state


@dataclass
class ProxyWorkRun:
    work_run: Callable
    work_name: str  # TODO: remove this argument and get the name from work.name directly
    work: "LightningWork"
    caller_queue: "BaseQueue"

    def __post_init__(self):
        self.work_state = None

    def __call__(self, *args, **kwargs):
        self.has_sent = False

        self._validate_call_args(args, kwargs)
        args, kwargs = self._process_call_args(args, kwargs)

        call_hash = self.work._call_hash(self.work_run, *self._convert_hashable(args, kwargs))
        entered = call_hash in self.work._calls
        returned = entered and "ret" in self.work._calls[call_hash]
        # TODO (tchaton): Handle spot instance retrieval differently from stopped work.
        stopped_on_sigterm = self.work._restarting and self.work.status.reason == WorkStopReasons.SIGTERM_SIGNAL_HANDLER

        data = {"args": args, "kwargs": kwargs, "call_hash": call_hash}

        # The if/else conditions are left un-compressed to simplify readability
        # for the readers.
        if self.work.cache_calls:
            if not entered or stopped_on_sigterm:
                _send_data_to_caller_queue(self, self.work, self.caller_queue, data, call_hash)
            else:
                if returned:
                    return
        else:
            if not entered or stopped_on_sigterm:
                _send_data_to_caller_queue(self, self.work, self.caller_queue, data, call_hash)
            else:
                if returned or stopped_on_sigterm:
                    # the previous task has completed and we can re-queue the next one.
                    # overriding the return value for next loop iteration.
                    _send_data_to_caller_queue(self, self.work, self.caller_queue, data, call_hash)
        if not self.work.parallel:
            raise CacheMissException("Task never called before. Triggered now")

    def _validate_call_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """Validate the call args before they get passed to the run method of the Work.

        Currently, this performs a check against strings that look like filesystem paths and may need to be wrapped with
        a Lightning Path by the user.
        """

        def warn_if_pathlike(obj: Union[os.PathLike, str]):
            if isinstance(obj, Path):
                return
            if os.sep in str(obj) and os.path.exists(obj):
                # NOTE: The existence check is wrong in general, as the file will never exist on the disk
                # where the flow is running unless we are running locally
                warnings.warn(
                    f"You passed a the value {obj!r} as an argument to the `run()` method of {self.work_name} and"
                    f" it looks like this is a path to a file or a folder. Consider wrapping this path in a"
                    f" `lightning_app.storage.Path` object to be able to access these files in your Work.",
                    UserWarning,
                )

        apply_to_collection(args, dtype=(os.PathLike, str), function=warn_if_pathlike)
        apply_to_collection(kwargs, dtype=(os.PathLike, str), function=warn_if_pathlike)

    @staticmethod
    def _process_call_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Processes all positional and keyword arguments before they get passed to the caller queue and sent to
        the LightningWork.

        Currently, this method only applies sanitization to Lightning Path objects.

        Args:
            args: The tuple of positional arguments passed to the run method.
            kwargs: The dictionary of named arguments passed to the run method.

        Returns:
            The positional and keyword arguments in the same order they were passed in.
        """

        def sanitize(obj: Union[Path, Drive]) -> Union[Path, Dict]:
            if isinstance(obj, Path):
                # create a copy of the Path and erase the consumer
                # the LightningWork on the receiving end of the caller queue will become the new consumer
                # this is necessary to make the Path deepdiff-hashable
                path_copy = Path(obj)
                path_copy._sanitize()
                path_copy._consumer = None
                return path_copy
            return obj.to_dict()

        return apply_to_collection((args, kwargs), dtype=(Path, Drive), function=sanitize)

    @staticmethod
    def _convert_hashable(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Processes all positional and keyword arguments before they get passed to the caller queue and sent to
        the LightningWork.

        Currently, this method only applies sanitization to Hashable Objects.

        Args:
            args: The tuple of positional arguments passed to the run method.
            kwargs: The dictionary of named arguments passed to the run method.

        Returns:
            The positional and keyword arguments in the same order they were passed in.
        """
        from lightning_app.utilities.types import Hashable

        def sanitize(obj: Hashable) -> Union[Path, Dict]:
            return obj.to_dict()

        return apply_to_collection((args, kwargs), dtype=Hashable, function=sanitize)


class WorkStateObserver(Thread):
    """This thread runs alongside LightningWork and periodically checks for state changes. If the state changed
    from one interval to the next, it will compute the delta and add it to the queue which is connected to the
    Flow. This enables state changes to be captured that are not triggered through a setattr call.

    Args:
        work: The LightningWork for which the state should be monitored
        delta_queue: The queue to send deltas to when state changes occur
        interval: The interval at which to check for state changes.

    Example:

        class Work(LightningWork):
            ...

            def run(self):
                # This update gets sent to the Flow once the thread compares the new state with the previous one
                self.list.append(1)
    """

    def __init__(
        self,
        work: "LightningWork",
        delta_queue: "BaseQueue",
        flow_to_work_delta_queue: Optional["BaseQueue"] = None,
        error_queue: Optional["BaseQueue"] = None,
        interval: float = 1,
    ) -> None:
        super().__init__(daemon=True)
        self.started = False
        self._work = work
        self._delta_queue = delta_queue
        self._flow_to_work_delta_queue = flow_to_work_delta_queue
        self._error_queue = error_queue
        self._interval = interval
        self._exit_event = Event()
        self._delta_memory = []
        self._last_state = deepcopy(self._work.state)

    def run(self) -> None:
        self.started = True
        while not self._exit_event.is_set():
            time.sleep(self._interval)
            # Run the thread only if active
            self.run_once()

    @staticmethod
    def get_state_changed_from_queue(q: "BaseQueue", timeout: Optional[int] = None):
        try:
            delta = q.get(timeout=timeout or q.default_timeout)
            return delta
        except queue.Empty:
            return None

    def run_once(self) -> None:
        with _state_observer_lock:
            # Add all deltas the LightningWorkSetAttrProxy has processed and sent to the Flow already while
            # the WorkStateObserver was sleeping
            for delta in self._delta_memory:
                self._last_state += delta
            self._delta_memory.clear()

            # The remaining delta is the result of state updates triggered outside the setattr, e.g, by a list append
            delta = Delta(DeepDiff(self._last_state, self._work.state, verbose_level=2))
            if not delta.to_dict():
                return
            self._last_state = deepcopy(self._work.state)
            self._delta_queue.put(ComponentDelta(id=self._work.name, delta=delta))

        if self._flow_to_work_delta_queue:
            while True:
                deep_diff = self.get_state_changed_from_queue(self._flow_to_work_delta_queue)
                if not isinstance(deep_diff, dict):
                    break
                try:
                    with _state_observer_lock:
                        self._work.apply_flow_delta(Delta(deep_diff, raise_errors=True))
                except Exception as e:
                    print(traceback.print_exc())
                    self._error_queue.put(e)
                    raise e

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)


@dataclass
class LightningWorkSetAttrProxy:
    """This wrapper around the ``LightningWork.__setattr__`` ensures that state changes get sent to the delta queue
    to be reflected in the Flow.

    Example:

        class Work(LightningWork):
            ...

            def run(self):
                self.var += 1  # This update gets sent to the Flow immediately
    """

    work_name: str
    work: "LightningWork"
    delta_queue: "BaseQueue"
    state_observer: Optional["WorkStateObserver"]

    def __call__(self, name: str, value: Any) -> None:
        logger.debug(f"Setting {name}: {value}")
        with _state_observer_lock:
            state = deepcopy(self.work.state)
            self.work._default_setattr(name, value)
            delta = Delta(DeepDiff(state, self.work.state, verbose_level=2))
            if not delta.to_dict():
                return

            # push the delta only if there is any
            self.delta_queue.put(ComponentDelta(id=self.work_name, delta=delta))

            # add the delta to the buffer to let WorkStateObserver know we already sent this one to the Flow
            if self.state_observer:
                self.state_observer._delta_memory.append(delta)


@dataclass
class ComponentDelta:
    id: str
    delta: Delta


@dataclass
class WorkRunExecutor:
    work: "LightningWork"
    work_run: Callable
    delta_queue: "BaseQueue"
    enable_start_observer: bool = True

    def __call__(self, *args, **kwargs):
        return self.work_run(*args, **kwargs)

    @contextmanager
    def enable_spawn(self) -> Generator:
        self.work._setattr_replacement = None
        self.work._backend = None
        self._clean_queues()
        yield

    def _clean_queues(self):
        if not isinstance(self.work._request_queue, MultiProcessQueue):
            self.work._request_queue = self.work._request_queue.to_dict()
            self.work._response_queue = self.work._response_queue.to_dict()

    @staticmethod
    def process_queue(queue):
        from lightning_app.core.queues import HTTPQueue, RedisQueue

        if isinstance(queue, dict):
            queue_type = queue.pop("type")
            if queue_type == "redis":
                return RedisQueue.from_dict(queue)
            else:
                return HTTPQueue.from_dict(queue)
        return queue


@dataclass
class WorkRunner:
    work: "LightningWork"
    work_name: str
    caller_queue: "BaseQueue"
    delta_queue: "BaseQueue"
    readiness_queue: "BaseQueue"
    error_queue: "BaseQueue"
    request_queue: "BaseQueue"
    response_queue: "BaseQueue"
    copy_request_queue: "BaseQueue"
    copy_response_queue: "BaseQueue"
    flow_to_work_delta_queue: Optional["BaseQueue"] = None
    run_executor_cls: Type[WorkRunExecutor] = WorkRunExecutor

    def __post_init__(self):
        self.parallel = self.work.parallel
        self.copier: Optional[_Copier] = None
        self.state_observer: Optional[WorkStateObserver] = None

    def __call__(self):
        self.setup()
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                if self.state_observer:
                    if self.state_observer.started:
                        self.state_observer.join(0)
                    self.state_observer = None
                self.copier.join(0)
            except LightningSigtermStateException as e:
                logger.debug("Exiting")
                os._exit(e.exit_code)
            except Exception as e:
                # Inform the flow the work failed. This would fail the entire application.
                self.error_queue.put(e)
                # Terminate the threads
                if self.state_observer:
                    if self.state_observer.started:
                        self.state_observer.join(0)
                    self.state_observer = None
                self.copier.join(0)
                raise e

    def setup(self):
        from lightning_app.utilities.state import AppState

        _set_work_context()

        # 1. Make the AppState aware of the affiliation of the work.
        # hacky: attach affiliation to be know from the AppState object
        AppState._MY_AFFILIATION = affiliation(self.work)

        # 2. Attach the queues to the work.
        # Each work gets their own request- and response storage queue for communicating with the storage orchestrator
        self.work._request_queue = self.request_queue
        self.work._response_queue = self.response_queue

        # 3. Starts the Copier thread. This thread enables transfering files using
        # the Path object between works.
        self.copier = _Copier(self.work, self.copy_request_queue, self.copy_response_queue)
        self.copier.setDaemon(True)
        self.copier.start()

        # 4. If the work is restarting, reload the latest state.
        # TODO (tchaton) Add support for capturing the latest state.
        if self.work._restarting:
            self.work.load_state_dict(self.work.state)

        # 5. Inform the flow that the work is ready to receive data through the caller queue.
        self.readiness_queue.put(True)

    def run_once(self):
        # 1. Wait for the caller queue data.
        called: Dict[str, Any] = self.caller_queue.get()
        logger.debug(f"Work {self.work_name} {called}")

        # 2. Extract the info from the caller queue data and process the input arguments. Arguments can contain
        # Lightning Path objects; if they don't have a consumer, the current Work will become one.
        call_hash = called["call_hash"]
        args, kwargs = self._process_call_args(called["args"], called["kwargs"])

        # 3. Register the signal handler for spot instances.
        # `SIGUSR1` signal isn't supported on windows.
        # TODO (tchaton) Add support for windows
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, partial(self._sigterm_signal_handler, call_hash=call_hash))

        # 4. Set the received state to the work.
        self.work.set_state(called["state"])

        # 5. Transfer all paths in the state automatically if they have an origin and exist
        self._transfer_path_attributes()

        # 6. Create the state observer thread.
        if self.run_executor_cls.enable_start_observer:
            self.state_observer = WorkStateObserver(
                self.work,
                delta_queue=self.delta_queue,
                flow_to_work_delta_queue=self.flow_to_work_delta_queue,
                error_queue=self.error_queue,
            )

        # 7. Deepcopy the work state and send the first `RUNNING` status delta to the flow.
        reference_state = deepcopy(self.work.state)

        # Set the internal IP address.
        # Set this here after the state observer is initialized, since it needs to record it as a change and send
        # it back to the flow
        default_internal_ip = "127.0.0.1" if constants.LIGHTNING_CLOUDSPACE_HOST is None else "0.0.0.0"
        self.work._internal_ip = os.environ.get("LIGHTNING_NODE_IP", default_internal_ip)

        # 8. Patch the setattr method of the work. This needs to be done after step 4, so we don't
        # send delta while calling `set_state`.
        self._proxy_setattr()

        if self._is_starting(called, reference_state, call_hash):
            return

        # 9. Inform the flow the work is running and add the delta to the deepcopy.
        self.work._calls[CacheCallsKeys.LATEST_CALL_HASH] = call_hash
        self.work._calls[call_hash]["statuses"].append(make_status(WorkStageStatus.RUNNING))
        delta = Delta(DeepDiff(reference_state, self.work.state))
        self.delta_queue.put(ComponentDelta(id=self.work_name, delta=delta))

        # 10. Unwrap the run method if wrapped.
        work_run = self.work.run
        if hasattr(work_run, "__wrapped__"):
            work_run = work_run.__wrapped__

        # 11. Start the state observer thread. It will look for state changes and send them back to the Flow
        # The observer has to be initialized here, after the set_state call above so that the thread can start with
        # the proper initial state of the work
        if self.run_executor_cls.enable_start_observer:
            self.state_observer.start()

        # 12. Run the `work_run` method.
        # If an exception is raised, send a `FAILED` status delta to the flow and call the `on_exception` hook.
        try:
            ret = self.run_executor_cls(self.work, work_run, self.delta_queue)(*args, **kwargs)
        except LightningSigtermStateException as e:
            raise e
        except BaseException as e:
            # 10.2 Send failed delta to the flow.
            reference_state = deepcopy(self.work.state)
            exp, val, tb = sys.exc_info()
            listing = traceback.format_exception(exp, val, tb)
            user_exception = False
            used_runpy = False
            trace = []
            for p in listing:
                if "runpy.py" in p:
                    trace = []
                    used_runpy = True
                if user_exception:
                    trace.append(p)
                if "ret = self.run_executor_cls(" in p:
                    user_exception = True

            if used_runpy:
                trace = trace[1:]

            self.work._calls[call_hash]["statuses"].append(
                make_status(
                    WorkStageStatus.FAILED,
                    message=str("\n".join(trace)),
                    reason=WorkFailureReasons.USER_EXCEPTION,
                )
            )
            self.delta_queue.put(
                ComponentDelta(
                    id=self.work_name, delta=Delta(DeepDiff(reference_state, self.work.state, verbose_level=2))
                )
            )
            self.work.on_exception(e)
            print("########## CAPTURED EXCEPTION ###########")
            print(traceback.print_exc())
            print("########## CAPTURED EXCEPTION ###########")
            return

        # 13. Destroy the state observer.
        if self.run_executor_cls.enable_start_observer:
            if self.state_observer.started:
                self.state_observer.join(0)
        self.state_observer = None

        # 14. Copy all artifacts to the shared storage so other Works can access them while this Work gets scaled down
        persist_artifacts(work=self.work)

        # 15. An asynchronous work shouldn't return a return value.
        if ret is not None:
            raise RuntimeError(
                f"Your work {self.work} shouldn't have a return value. Found {ret}."
                "HINT: Use the Payload API instead."
            )

        # 17. DeepCopy the state and send the latest delta to the flow.
        # use the latest state as we have already sent delta
        # during its execution.
        # inform the task has completed
        reference_state = deepcopy(self.work.state)
        self.work._calls[call_hash]["statuses"].append(make_status(WorkStageStatus.SUCCEEDED))
        self.work._calls[call_hash]["ret"] = ret
        self.delta_queue.put(
            ComponentDelta(id=self.work_name, delta=Delta(DeepDiff(reference_state, self.work.state, verbose_level=2)))
        )

        # 18. Update the work for the next delta if any.
        self._proxy_setattr(cleanup=True)

    def _sigterm_signal_handler(self, signum, frame, call_hash: str) -> None:
        """Signal handler used to react when spot instances are being retrived."""
        logger.info(f"Received SIGTERM signal. Gracefully terminating {self.work.name.replace('root.', '')}...")
        persist_artifacts(work=self.work)
        with _state_observer_lock:
            self.work.on_exit()
            self.work._calls[call_hash]["statuses"] = []
            state = deepcopy(self.work.state)
            self.work._calls[call_hash]["statuses"].append(
                make_status(WorkStageStatus.STOPPED, reason=WorkStopReasons.SIGTERM_SIGNAL_HANDLER)
            )

        # kill the thread as the job is going to be terminated.
        if self.state_observer:
            if self.state_observer.started:
                self.state_observer.join(0)
            self.state_observer = None
        delta = Delta(DeepDiff(state, deepcopy(self.work.state), verbose_level=2))
        self.delta_queue.put(ComponentDelta(id=self.work_name, delta=delta))

        self.copier.join(0)
        raise LightningSigtermStateException(0)

    def _proxy_setattr(self, cleanup: bool = False):
        _proxy_setattr(self.work, self.delta_queue, self.state_observer, cleanup=cleanup)

    def _process_call_args(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Process the arguments that were passed in to the ``run()`` method of the
        :class:`lightning_app.core.work.LightningWork`.

        This method currently only implements special treatments for the :class:`lightning_app.storage.path.Path`
        objects. Any Path objects that get passed into the run method get attached to the Work automatically, i.e.,
        the Work becomes the `origin` or the `consumer` if they were not already before. Additionally,
        if the file or folder under the Path exists, we transfer it.

        Args:
            args: The tuple of positional arguments passed to the run method.
            kwargs: The dictionary of named arguments passed to the run method.

        Returns:
            The positional and keyword arguments in the same order they were passed in.
        """

        def _attach_work_and_get(transporter: Union[Path, Payload, dict]) -> Union[Path, Drive, dict, Any]:
            if not transporter.origin_name:
                # If path/payload is not attached to an origin, there is no need to attach or transfer anything
                return transporter

            transporter._attach_work(self.work)
            transporter._attach_queues(self.work._request_queue, self.work._response_queue)
            if transporter.exists_remote():
                # All paths/payloads passed to the `run` method under a Lightning obj need to be copied (if they exist)
                if isinstance(transporter, Payload):
                    transporter.get()
                else:
                    transporter.get(overwrite=True)
            return transporter

        def _handle_drive(dict):
            return _maybe_create_drive(self.work_name, dict)

        args, kwargs = apply_to_collection((args, kwargs), dtype=(Path, Payload), function=_attach_work_and_get)
        return apply_to_collection((args, kwargs), dtype=dict, function=_handle_drive)

    def _transfer_path_attributes(self) -> None:
        """Transfer all Path attributes in the Work if they have an origin and exist."""
        for name in self.work._paths:
            path = getattr(self.work, name)
            if isinstance(path, str):
                path = Path(path)
                path._attach_work(self.work)
            if path.origin_name and path.origin_name != self.work.name and path.exists_remote():
                path.get(overwrite=True)

    def _is_starting(self, called, reference_state, call_hash) -> bool:
        if len(called["args"]) == 1 and isinstance(called["args"][0], Action):
            action = called["args"][0]
            if action.method == "start":
                # 9. Inform the flow the work is running and add the delta to the deepcopy.
                self.work._calls[CacheCallsKeys.LATEST_CALL_HASH] = call_hash
                self.work._calls[call_hash]["statuses"].append(make_status(WorkStageStatus.STARTED))
                delta = Delta(DeepDiff(reference_state, self.work.state))
                self.delta_queue.put(ComponentDelta(id=self.work_name, delta=delta))
                self._proxy_setattr(cleanup=True)
                return True
            else:
                raise Exception("Only the `start` action is supported right now !")
        return False


def persist_artifacts(work: "LightningWork") -> None:
    """Copies all :class:`~lightning_app.storage.path.Path` referenced by the given LightningWork to the shared
    storage.

    Files that don't exist or do not originate from the given Work will be skipped.
    """
    artifact_paths = [getattr(work, name) for name in work._paths]
    # only copy files that belong to this Work, i.e., when the path's origin refers to the current Work
    artifact_paths = [path for path in artifact_paths if isinstance(path, Path) and path.origin_name == work.name]

    for name in work._state:
        if isinstance(getattr(work, name), Payload):
            artifact_path = pathlib.Path(name).resolve()
            payload = getattr(work, name)
            payload.save(payload.value, artifact_path)
            artifact_paths.append(artifact_path)

    missing_artifacts: Set[str] = set()
    destination_paths = []
    for artifact_path in artifact_paths:
        artifact_path = pathlib.Path(artifact_path).absolute()
        if not artifact_path.exists():
            missing_artifacts.add(str(artifact_path))
            continue
        destination_path = _path_to_work_artifact(artifact_path, work)
        _copy_files(artifact_path, destination_path)
        destination_paths.append(destination_path)

    if missing_artifacts:
        warnings.warn(
            f"{len(missing_artifacts)} artifacts could not be saved because they don't exist:"
            f" {','.join(missing_artifacts)}.",
            UserWarning,
        )
    else:
        logger.debug(
            f"All {destination_paths} artifacts from Work {work.name} successfully "
            "stored at {artifacts_path(work.name)}."
        )


def _proxy_setattr(work, delta_queue, state_observer: Optional[WorkStateObserver], cleanup: bool = False):
    if cleanup:
        setattr_proxy = None
    else:
        setattr_proxy = LightningWorkSetAttrProxy(
            work.name,
            work,
            delta_queue=delta_queue,
            state_observer=state_observer,
        )
    work._setattr_replacement = setattr_proxy
