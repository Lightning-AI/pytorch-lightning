import logging
import os
import pathlib
import signal
import sys
import threading
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

from deepdiff import DeepDiff, Delta

from lightning_app.storage import Path
from lightning_app.storage.copier import Copier, copy_files
from lightning_app.storage.drive import _maybe_create_drive, Drive
from lightning_app.storage.path import path_to_work_artifact
from lightning_app.storage.payload import Payload
from lightning_app.utilities.app_helpers import affiliation
from lightning_app.utilities.apply_func import apply_to_collection
from lightning_app.utilities.component import _set_work_context
from lightning_app.utilities.enum import make_status, WorkFailureReasons, WorkStageStatus, WorkStopReasons
from lightning_app.utilities.exceptions import CacheMissException, LightningSigtermStateException

if TYPE_CHECKING:
    from lightning_app import LightningWork
    from lightning_app.core.queues import BaseQueue


logger = logging.getLogger(__name__)
_state_observer_lock = threading.Lock()


def unwrap(fn):
    if isinstance(fn, partial):
        fn = fn.keywords["work_run"]
    if isinstance(fn, ProxyWorkRun):
        fn = fn.work_run
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _send_data_to_caller_queue(
    work: "LightningWork", caller_queue: "BaseQueue", data: Dict, call_hash: str, work_run: Callable, use_args: bool
) -> Dict:
    if work._calls["latest_call_hash"] is None:
        work._calls["latest_call_hash"] = call_hash

    if call_hash not in work._calls:
        work._calls[call_hash] = {
            "name": work_run.__name__,
            "call_hash": call_hash,
            "use_args": use_args,
            "statuses": [],
        }
    else:
        # remove ret when relaunching the work.
        work._calls[call_hash].pop("ret", None)

    work._calls[call_hash]["statuses"].append(make_status(WorkStageStatus.PENDING))

    work_state = work.state
    data.update({"state": work_state})
    logger.debug(f"Sending to {work.name}: {data}")
    caller_queue.put(data)
    work._restarting = False
    return work_state


@dataclass
class ProxyWorkRun:
    work_run: Callable
    work_name: str  # TODO: remove this argument and get the name from work.name directly
    work: "LightningWork"
    caller_queue: "BaseQueue"

    def __post_init__(self):
        self.cache_calls = self.work.cache_calls
        self.parallel = self.work.parallel
        self.work_state = None

    def __call__(self, *args, **kwargs):
        provided_none = len(args) == 1 and args[0] is None
        use_args = len(kwargs) > 0 or (len(args) > 0 and not provided_none)

        self._validate_call_args(args, kwargs)
        args, kwargs = self._process_call_args(args, kwargs)

        call_hash = self.work._call_hash(self.work_run, args, kwargs)
        entered = call_hash in self.work._calls
        returned = entered and "ret" in self.work._calls[call_hash]
        # TODO (tchaton): Handle spot instance retrieval differently from stopped work.
        stopped_on_sigterm = self.work._restarting and self.work.status.reason == WorkStopReasons.SIGTERM_SIGNAL_HANDLER

        data = {"args": args, "kwargs": kwargs, "call_hash": call_hash}

        # The if/else conditions are left un-compressed to simplify readability
        # for the readers.
        if self.cache_calls:
            if not entered or stopped_on_sigterm:
                _send_data_to_caller_queue(self.work, self.caller_queue, data, call_hash, self.work_run, use_args)
            else:
                if returned:
                    return
        else:
            if not entered or stopped_on_sigterm:
                _send_data_to_caller_queue(self.work, self.caller_queue, data, call_hash, self.work_run, use_args)
            else:
                if returned or stopped_on_sigterm:
                    # the previous task has completed and we can re-queue the next one.
                    # overriding the return value for next loop iteration.
                    _send_data_to_caller_queue(self.work, self.caller_queue, data, call_hash, self.work_run, use_args)
        if not self.parallel:
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
        """Processes all positional and keyword arguments before they get passed to the caller queue and sent to the
        LightningWork.

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


class WorkStateObserver(Thread):
    """This thread runs alongside LightningWork and periodically checks for state changes.

    If the state changed from one interval to the next, it will compute the delta and add it to the queue which is
    connected to the Flow. This enables state changes to be captured that are not triggered through a setattr call.

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

    def __init__(self, work: "LightningWork", delta_queue: "BaseQueue", interval: float = 5) -> None:
        super().__init__(daemon=True)
        self._work = work
        self._delta_queue = delta_queue
        self._interval = interval
        self._exit_event = Event()
        self._delta_memory = []
        self._last_state = deepcopy(self._work.state)

    def run(self) -> None:
        while not self._exit_event.is_set():
            time.sleep(self._interval)
            # Run the thread only if active
            self.run_once()

    def run_once(self) -> None:
        with _state_observer_lock:
            # Add all deltas the LightningWorkSetAttrProxy has processed and sent to the Flow already while
            # the WorkStateObserver was sleeping
            for delta in self._delta_memory:
                self._last_state += delta
            self._delta_memory.clear()

            # The remaining delta is the result of state updates triggered outside the setattr, e.g, by a list append
            delta = Delta(DeepDiff(self._last_state, self._work.state))
            if not delta.to_dict():
                return
            self._last_state = deepcopy(self._work.state)
            self._delta_queue.put(ComponentDelta(id=self._work.name, delta=delta))

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)


@dataclass
class LightningWorkSetAttrProxy:
    """This wrapper around the ``LightningWork.__setattr__`` ensures that state changes get sent to the delta queue to
    be reflected in the Flow.

    Example:

        class Work(LightningWork):
            ...

            def run(self):
                self.var += 1  # This update gets sent to the Flow immediately
    """

    work_name: str
    work: "LightningWork"
    delta_queue: "BaseQueue"
    state_observer: "WorkStateObserver"

    def __call__(self, name: str, value: Any) -> None:
        logger.debug(f"Setting {name}: {value}")
        with _state_observer_lock:
            state = deepcopy(self.work.state)
            self.work._default_setattr(name, value)
            delta = Delta(DeepDiff(state, self.work.state))
            if not delta.to_dict():
                return

            # push the delta only if there is any
            self.delta_queue.put(ComponentDelta(id=self.work_name, delta=delta))

            # add the delta to the buffer to let WorkStateObserver know we already sent this one to the Flow
            self.state_observer._delta_memory.append(delta)


@dataclass
class ComponentDelta:
    id: str
    delta: Delta


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

    def __post_init__(self):
        self.parallel = self.work.parallel
        self.copier: Optional[Copier] = None
        self.state_observer: Optional[WorkStateObserver] = None

    def __call__(self):
        self.setup()
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                if self.state_observer:
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
        self.copier = Copier(self.work, self.copy_request_queue, self.copy_response_queue)
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
        self.state_observer = WorkStateObserver(self.work, delta_queue=self.delta_queue)

        # Set the internal IP address.
        # Set this here after the state observer is initialized, since it needs to record it as a change and send
        # it back to the flow
        self.work._internal_ip = os.environ.get("LIGHTNING_NODE_IP", "127.0.0.1")

        # 7. Patch the setattr method of the work. This needs to be done after step 4, so we don't
        # send delta while calling `set_state`.
        self._proxy_setattr()

        # 8. Deepcopy the work state and send the first `RUNNING` status delta to the flow.
        state = deepcopy(self.work.state)
        self.work._calls["latest_call_hash"] = call_hash
        self.work._calls[call_hash]["statuses"].append(make_status(WorkStageStatus.RUNNING))
        self.delta_queue.put(ComponentDelta(id=self.work_name, delta=Delta(DeepDiff(state, self.work.state))))

        # 9. Start the state observer thread. It will look for state changes and send them back to the Flow
        # The observer has to be initialized here, after the set_state call above so that the thread can start with
        # the proper initial state of the work
        self.state_observer.start()

        # 10. Unwrap the run method if wrapped.
        work_run = self.work.run
        if hasattr(work_run, "__wrapped__"):
            work_run = work_run.__wrapped__

        # 12. Run the `work_run` method.
        # If an exception is raised, send a `FAILED` status delta to the flow and call the `on_exception` hook.
        try:
            ret = work_run(*args, **kwargs)
        except LightningSigtermStateException as e:
            raise e
        except BaseException as e:
            # 10.2 Send failed delta to the flow.
            self.work._calls[call_hash]["statuses"].append(
                make_status(WorkStageStatus.FAILED, message=str(e), reason=WorkFailureReasons.USER_EXCEPTION)
            )
            self.delta_queue.put(ComponentDelta(id=self.work_name, delta=Delta(DeepDiff(state, self.work.state))))
            self.work.on_exception(e)
            return

        # 14. Copy all artifacts to the shared storage so other Works can access them while this Work gets scaled down
        persist_artifacts(work=self.work)

        # 15. Destroy the state observer.
        self.state_observer.join(0)
        self.state_observer = None

        # 15. An asynchronous work shouldn't return a return value.
        if ret is not None:
            raise RuntimeError(
                f"Your work {self.work} shouldn't have a return value. Found {ret}."
                "HINT: Use the Payload API instead."
            )

        # 16. DeepCopy the state and send the latest delta to the flow.
        # use the latest state as we have already sent delta
        # during its execution.
        # inform the task has completed
        state = deepcopy(self.work.state)
        self.work._calls[call_hash]["statuses"].append(make_status(WorkStageStatus.SUCCEEDED))
        self.work._calls[call_hash]["ret"] = ret
        self.delta_queue.put(ComponentDelta(id=self.work_name, delta=Delta(DeepDiff(state, self.work.state))))

        # 17. Update the work for the next delta if any.
        self._proxy_setattr(cleanup=True)

    def _sigterm_signal_handler(self, signum, frame, call_hash: str) -> None:
        """Signal handler used to react when spot instances are being retrived."""
        logger.debug("Received SIGTERM signal. Gracefully terminating...")
        persist_artifacts(work=self.work)
        with _state_observer_lock:
            state = deepcopy(self.work.state)
            self.work._calls[call_hash]["statuses"].append(
                make_status(WorkStageStatus.STOPPED, reason=WorkStopReasons.SIGTERM_SIGNAL_HANDLER)
            )
            delta = Delta(DeepDiff(state, self.work.state))
            self.delta_queue.put(ComponentDelta(id=self.work_name, delta=delta))

        # kill the thread as the job is going to be terminated.
        self.copier.join(0)
        if self.state_observer:
            self.state_observer.join(0)
            self.state_observer = None
        raise LightningSigtermStateException(0)

    def _proxy_setattr(self, cleanup: bool = False):
        if cleanup:
            setattr_proxy = None
        else:
            assert self.state_observer
            setattr_proxy = LightningWorkSetAttrProxy(
                self.work_name, self.work, delta_queue=self.delta_queue, state_observer=self.state_observer
            )
        self.work._setattr_replacement = setattr_proxy

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


def persist_artifacts(work: "LightningWork") -> None:
    """Copies all :class:`~lightning_app.storage.path.Path` referenced by the given LightningWork to the shared storage.

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
        destination_path = path_to_work_artifact(artifact_path, work)
        copy_files(artifact_path, destination_path)
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
