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

import logging
import os
import pickle
import queue
import threading
import warnings
from copy import deepcopy
from time import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from deepdiff import DeepDiff, Delta
from lightning_utilities.core.apply_func import apply_to_collection

import lightning_app
from lightning_app import _console
from lightning_app.api.request_types import _APIRequest, _CommandRequest, _DeltaRequest
from lightning_app.core.constants import (
    DEBUG_ENABLED,
    FLOW_DURATION_SAMPLES,
    FLOW_DURATION_THRESHOLD,
    FRONTEND_DIR,
    STATE_ACCUMULATE_WAIT,
)
from lightning_app.core.queues import BaseQueue
from lightning_app.core.work import LightningWork
from lightning_app.frontend import Frontend
from lightning_app.storage import Drive, Path, Payload
from lightning_app.storage.path import _storage_root_dir
from lightning_app.utilities import frontend
from lightning_app.utilities.app_helpers import (
    _delta_to_app_state_delta,
    _handle_is_headless,
    _is_headless,
    _LightningAppRef,
    _should_dispatch_app,
    Logger,
)
from lightning_app.utilities.app_status import AppStatus
from lightning_app.utilities.commands.base import _process_requests
from lightning_app.utilities.component import _convert_paths_after_init, _validate_root_flow
from lightning_app.utilities.enum import AppStage, CacheCallsKeys
from lightning_app.utilities.exceptions import CacheMissException, ExitAppException
from lightning_app.utilities.layout import _collect_layout
from lightning_app.utilities.proxies import ComponentDelta
from lightning_app.utilities.scheduler import SchedulerThread
from lightning_app.utilities.tree import breadth_first
from lightning_app.utilities.warnings import LightningFlowWarning

if TYPE_CHECKING:
    from lightning_app.core.flow import LightningFlow
    from lightning_app.runners.backends.backend import Backend, WorkManager
    from lightning_app.runners.runtime import Runtime


logger = Logger(__name__)


class LightningApp:
    def __init__(
        self,
        root: Union["LightningFlow", "LightningWork"],
        flow_cloud_compute: Optional["lightning_app.CloudCompute"] = None,
        log_level: str = "info",
        info: frontend.AppInfo = None,
        root_path: str = "",
    ):
        """The Lightning App, or App in short runs a tree of one or more components that interact to create end-to-end
        applications. There are two kinds of components: :class:`~lightning_app.core.flow.LightningFlow` and
        :class:`~lightning_app.core.work.LightningWork`. This modular design enables you to reuse components
        created by other users.

        The Lightning App alternatively run an event loop triggered by delta changes sent from
        either :class:`~lightning_app.core.work.LightningWork` or from the Lightning UI.
        Once deltas are received, the Lightning App runs
        the :class:`~lightning_app.core.flow.LightningFlow` provided.

        Arguments:
            root: The root ``LightningFlow`` or ``LightningWork`` component, that defines all the app's nested
                 components, running infinitely. It must define a `run()` method that the app can call.
            flow_cloud_compute: The default Cloud Compute used for flow, Rest API and frontend's.
            log_level: The log level for the app, one of [`info`, `debug`].
                This can be helpful when reporting bugs on Lightning repo.
            info: Provide additional info about the app which will be used to update html title,
                description and image meta tags and specify any additional tags as list of html strings.
            root_path: Set this to `/path` if you want to run your app behind a proxy at `/path` leave empty for "/".
                For instance, if you want to run your app at `https://customdomain.com/myapp`,
                set `root_path` to `/myapp`.
                You can learn more about proxy `here <https://www.fortinet.com/resources/cyberglossary/proxy-server>`_.


        Example:

            >>> from lightning_app import LightningFlow, LightningApp
            >>> from lightning_app.runners import MultiProcessRuntime
            >>> class RootFlow(LightningFlow):
            ...     def run(self):
            ...         self.stop()
            ...
            >>> app = LightningApp(RootFlow())  # application can be dispatched using the `runners`.
            >>> MultiProcessRuntime(app).dispatch()
        """

        self.root_path = root_path  # when running behind a proxy
        self.info = info

        from lightning_app.core.flow import _RootFlow

        if isinstance(root, LightningWork):
            root = _RootFlow(root)

        _validate_root_flow(root)
        self._root = root
        self.flow_cloud_compute = flow_cloud_compute or lightning_app.CloudCompute(name="flow-lite")

        # queues definition.
        self.delta_queue: Optional[BaseQueue] = None
        self.readiness_queue: Optional[BaseQueue] = None
        self.api_response_queue: Optional[BaseQueue] = None
        self.api_publish_state_queue: Optional[BaseQueue] = None
        self.api_delta_queue: Optional[BaseQueue] = None
        self.error_queue: Optional[BaseQueue] = None
        self.request_queues: Optional[Dict[str, BaseQueue]] = None
        self.response_queues: Optional[Dict[str, BaseQueue]] = None
        self.copy_request_queues: Optional[Dict[str, BaseQueue]] = None
        self.copy_response_queues: Optional[Dict[str, BaseQueue]] = None
        self.caller_queues: Optional[Dict[str, BaseQueue]] = None
        self.flow_to_work_delta_queues: Optional[Dict[str, BaseQueue]] = None
        self.work_queues: Optional[Dict[str, BaseQueue]] = None
        self.commands: Optional[List] = None

        self.should_publish_changes_to_api = False
        self.component_affiliation = None
        self.backend: Optional["Backend"] = None
        _LightningAppRef.connect(self)
        self.processes: Dict[str, "WorkManager"] = {}
        self.frontends: Dict[str, Frontend] = {}
        self.stage = AppStage.RUNNING
        self._has_updated: bool = True
        self._schedules: Dict[str, Dict] = {}
        self.threads: List[threading.Thread] = []
        self.exception = None
        self.collect_changes: bool = True

        self.status: Optional[AppStatus] = None
        # TODO: Enable ready locally for opening the UI.
        self.ready = False

        # NOTE: Checkpointing is disabled by default for the time being.  We
        # will enable it when resuming from full checkpoint is supported. Also,
        # we will need to revisit the logic at _should_snapshot, since right now
        # we are writing checkpoints too often, and this is expensive.
        self.checkpointing: bool = False

        self._update_layout()
        self._update_status()

        self.is_headless: Optional[bool] = None

        self._original_state = None
        self._last_state = self.state
        self.state_accumulate_wait = STATE_ACCUMULATE_WAIT

        self._last_run_time = 0.0
        self._run_times = []

        # Path attributes can't get properly attached during the initialization, because the full name
        # is only available after all Flows and Works have been instantiated.
        _convert_paths_after_init(self.root)

        if log_level not in ("debug", "info"):
            raise Exception(f"Log Level should be in ['debug', 'info']. Found {log_level}")

        # Lazily enable debugging.
        if log_level == "debug" or DEBUG_ENABLED:
            if not DEBUG_ENABLED:
                os.environ["LIGHTNING_DEBUG"] = "2"
            _console.setLevel(logging.DEBUG)

        logger.debug(f"ENV: {os.environ}")

        if _should_dispatch_app():
            os.environ["LIGHTNING_DISPATCHED"] = "1"
            from lightning_app.runners import MultiProcessRuntime

            MultiProcessRuntime(self).dispatch()

    def _update_index_file(self):
        # update index.html,
        # this should happen once for all apps before the ui server starts running.
        frontend.update_index_file(FRONTEND_DIR, info=self.info, root_path=self.root_path)

    def get_component_by_name(self, component_name: str):
        """Returns the instance corresponding to the given component name."""
        from lightning_app.structures import Dict as LightningDict
        from lightning_app.structures import List as LightningList
        from lightning_app.utilities.types import ComponentTuple

        if component_name == "root":
            return self.root
        if not component_name.startswith("root."):
            raise ValueError(f"Invalid component name {component_name}. Name must start with 'root'")

        current = self.root
        for child_name in component_name.split(".")[1:]:
            if isinstance(current, (LightningDict, LightningList)):
                child = current[child_name] if isinstance(current, LightningDict) else current[int(child_name)]
            else:
                child = getattr(current, child_name, None)
            if not isinstance(child, ComponentTuple):
                raise AttributeError(f"Component '{current.name}' has no child component with name '{child_name}'.")
            current = child
        return current

    def _reset_original_state(self):
        self.set_state(self._original_state)

    @property
    def root(self):
        """Returns the root component of the application."""
        return self._root

    @property
    def state(self):
        """Return the current state of the application."""
        state = self.root.state
        state["app_state"] = {"stage": self.stage.value}
        return state

    @property
    def state_vars(self):
        """Return the current state restricted to the user defined variables of the application."""
        state_vars = self.root.state_vars
        state_vars["app_state"] = {"stage": self.stage.value}
        return state_vars

    @property
    def state_with_changes(self):
        """Return the current state with the new changes of the application."""
        state_with_changes = self.root.state_with_changes
        state_with_changes["app_state"] = {"stage": self.stage.value}
        return state_with_changes

    def set_state(self, state):
        """Method to set a new app state set to the application."""
        self.set_last_state(state)
        self.root.set_state(state)
        self.stage = AppStage(state["app_state"]["stage"])

    @property
    def last_state(self):
        """Returns the latest state."""
        return self._last_state

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(_storage_root_dir(), "checkpoints")

    def remove_changes_(self, state):
        for _, child in state["flows"].items():
            self.remove_changes(child)
        state["changes"] = {}

    def remove_changes(self, state):
        state = deepcopy(state)
        for _, child in state["flows"].items():
            self.remove_changes_(child)
        state["changes"] = {}
        return state

    def set_last_state(self, state):
        self._last_state = self.remove_changes(state)

    @staticmethod
    def populate_changes(last_state, new_state):
        diff = DeepDiff(last_state, new_state, view="tree", verbose_level=2)

        changes_categories = [diff[key] for key in diff.to_dict()]

        if not changes_categories:
            return new_state

        for change_category in changes_categories:
            for entry in change_category:
                state_el = new_state
                change = entry.path(output_format="list")
                if "vars" not in change:
                    continue
                for change_el in change:
                    if change_el == "vars":
                        if "changes" not in state_el:
                            state_el["changes"] = {}
                        state_el["changes"][change[-1]] = {"from": entry.t1, "to": entry.t2}
                        break
                    # move down in the dictionary
                    state_el = state_el[change_el]
        return new_state

    @staticmethod
    def get_state_changed_from_queue(q: BaseQueue, timeout: Optional[int] = None):
        try:
            delta = q.get(timeout=timeout or q.default_timeout)
            return delta
        except queue.Empty:
            return None

    def check_error_queue(self) -> None:
        exception: Exception = self.get_state_changed_from_queue(self.error_queue)
        if isinstance(exception, Exception):
            self.exception = exception
            self.stage = AppStage.FAILED

    @property
    def flows(self) -> List["LightningFlow"]:
        """Returns all the flows defined within this application."""
        return [self.root] + list(self.root.flows.values())

    @property
    def works(self) -> List[LightningWork]:
        """Returns all the works defined within this application."""
        return self.root.works(recurse=True)

    @property
    def named_works(self) -> List[Tuple[str, LightningWork]]:
        """Returns all the works defined within this application with their names."""
        return self.root.named_works(recurse=True)

    def _collect_deltas_from_ui_and_work_queues(self) -> List[Union[Delta, _APIRequest, _CommandRequest]]:
        # The aggregation would try to get as many deltas as possible
        # from both the `api_delta_queue` and `delta_queue`
        # during the `state_accumulate_wait` time.
        # The while loop can exit sooner if both queues are empty.

        deltas = []
        api_or_command_request_deltas = []
        t0 = time()

        while (time() - t0) < self.state_accumulate_wait:

            # TODO: Fetch all available deltas at once to reduce queue calls.
            delta: Optional[
                Union[_DeltaRequest, _APIRequest, _CommandRequest, ComponentDelta]
            ] = self.get_state_changed_from_queue(self.delta_queue)
            if delta:
                if isinstance(delta, _DeltaRequest):
                    deltas.append(delta.delta)
                elif isinstance(delta, ComponentDelta):
                    logger.debug(f"Received from {delta.id} : {delta.delta.to_dict()}")
                    work = None
                    try:
                        work = self.get_component_by_name(delta.id)
                    except (KeyError, AttributeError) as e:
                        logger.error(f"The component {delta.id} couldn't be accessed. Exception: {e}")

                    if work:
                        delta = _delta_to_app_state_delta(self.root, work, deepcopy(delta.delta))
                        deltas.append(delta)
                else:
                    api_or_command_request_deltas.append(delta)
            else:
                break

        if api_or_command_request_deltas:
            _process_requests(self, api_or_command_request_deltas)

        for delta in deltas:
            # When aggregating deltas from the UI and the Works, and over the accumulation time window,
            # it can happen that deltas from these different sources disagree. Since deltas are computed on the Work
            # and UI side separately, correctness of the aggregation can only be guaranteed if both components compute
            # the delta based on the same base state. But this assumption does not hold in general, and there is no way
            # for the Flow to reject or resolve these deltas properly at the moment. Hence, we decide to ignore
            # errors coming from deepdiff when adding deltas together by setting:
            delta.log_errors = False
            delta.raise_errors = False
        return deltas

    def maybe_apply_changes(self) -> None:
        """Get the deltas from both the flow queue and the work queue, merge the two deltas and update the
        state."""
        self._send_flow_to_work_deltas(self.state)

        if not self.collect_changes:
            return None

        deltas = self._collect_deltas_from_ui_and_work_queues()

        if not deltas:
            # Path and Drive aren't processed by DeepDiff, so we need to convert them to dict.
            last_state = apply_to_collection(self.last_state, (Path, Drive), lambda x: x.to_dict())
            state = apply_to_collection(self.state, (Path, Drive), lambda x: x.to_dict())
            # When no deltas are received from the Rest API or work queues,
            # we need to check if the flow modified the state and populate changes.
            deep_diff = DeepDiff(last_state, state, verbose_level=2)

            if "unprocessed" in deep_diff:
                # pop the unprocessed key.
                unprocessed = deep_diff.pop("unprocessed")
                logger.warn(f"It seems delta differentiation resulted in {unprocessed}. Open an issue on Github.")

            if deep_diff:
                # TODO: Resolve changes with ``CacheMissException``.
                # new_state = self.populate_changes(self.last_state, self.state)
                self.set_last_state(self.state)
                self._has_updated = True
            return False

        logger.debug(f"Received {[d.to_dict() for d in deltas]}")

        # 2: Collect the state
        state = self.state

        # 3: Apply the state delta
        for delta in deltas:
            try:
                state += delta
            except Exception as e:
                raise Exception(f"Current State {state}, {delta.to_dict()}") from e

        # new_state = self.populate_changes(self.last_state, state)
        self.set_state(state)
        self._has_updated = True

    def run_once(self):
        """Method used to collect changes and run the root Flow once."""
        done = False
        self._last_run_time = 0.0

        if self.backend is not None:
            self.backend.update_work_statuses(self.works)

        self._update_layout()
        self._update_is_headless()
        self._update_status()
        self.maybe_apply_changes()

        if self.checkpointing and self._should_snapshot():
            self._dump_checkpoint()

        if self.stage == AppStage.BLOCKING:
            return done

        if self.stage in (AppStage.STOPPING, AppStage.FAILED):
            return True

        elif self.stage == AppStage.RESTARTING:
            return self._apply_restarting()

        t0 = time()

        try:
            self.check_error_queue()
            # Execute the flow only if:
            # - There are state changes
            # - It is the first execution of the flow
            if self._has_updated:
                self.root.run()
        except CacheMissException:
            self._on_cache_miss_exception()
        except (ExitAppException, KeyboardInterrupt):
            done = True
            self.stage = AppStage.STOPPING

        if not self.ready:
            self.ready = self.root.ready

        self._last_run_time = time() - t0

        self.on_run_once_end()
        return done

    def _reset_run_time_monitor(self):
        self._run_times = [0.0] * FLOW_DURATION_SAMPLES

    def _update_run_time_monitor(self):
        self._run_times[:-1] = self._run_times[1:]
        self._run_times[-1] = self._last_run_time

        # Here we underestimate during the first FLOW_DURATION_SAMPLES
        # iterations, but that's ok for our purposes
        avg_elapsed_time = sum(self._run_times) / FLOW_DURATION_SAMPLES

        if avg_elapsed_time > FLOW_DURATION_THRESHOLD:
            warnings.warn(
                "The execution of the `run` method of the root flow is taking too long. "
                "Flow is supposed to only host coordination logic, while currently it is"
                "likely to contain long-running calls, code that performs meaningful "
                "computations or that makes blocking or asynchronous calls to third-party "
                "services. If that is the case, you should move those pieces to a Work, "
                "and make sure Flow can complete its execution in under a second.",
                LightningFlowWarning,
            )

    def _run(self) -> bool:
        """Entry point of the LightningApp.

        This would be dispatched by the Runtime objects.
        """
        self._original_state = deepcopy(self.state)
        done = False

        self.ready = self.root.ready

        self._start_with_flow_works()

        if self.should_publish_changes_to_api and self.api_publish_state_queue is not None:
            self.api_publish_state_queue.put((self.state_vars, self.status))

        self._reset_run_time_monitor()

        while not done:
            done = self.run_once()

            self._update_run_time_monitor()

            if self._has_updated and self.should_publish_changes_to_api and self.api_publish_state_queue is not None:
                self.api_publish_state_queue.put((self.state_vars, self.status))

            self._has_updated = False

        self._on_run_end()

        return True

    def _update_layout(self) -> None:
        import lightning_app

        if self.backend:
            self.backend.resolve_url(self, base_url=None)

        for component in breadth_first(self.root, types=(lightning_app.LightningFlow,)):
            layout = _collect_layout(self, component)
            component._layout = layout

    def _update_is_headless(self) -> None:
        self.is_headless = _is_headless(self)

        # If `is_headless` changed, handle it.
        # This ensures support for apps which dynamically add a UI at runtime.
        _handle_is_headless(self)

    def _update_status(self) -> None:
        old_status = self.status

        work_statuses = {}
        for work in breadth_first(self.root, types=(lightning_app.LightningWork,)):
            work_statuses[work.name] = work.status

        self.status = AppStatus(
            is_ui_ready=self.ready,
            work_statuses=work_statuses,
        )

        # If the work statuses changed, the state delta will trigger an update.
        # If ready has changed, we trigger an update manually.
        if self.status != old_status:
            self._has_updated = True

    def _apply_restarting(self) -> bool:
        self._reset_original_state()
        # apply stage after restoring the original state.
        self.stage = AppStage.BLOCKING
        return False

    def _has_work_finished(self, work) -> bool:
        latest_call_hash = work._calls[CacheCallsKeys.LATEST_CALL_HASH]
        if latest_call_hash is None:
            return False
        return "ret" in work._calls[latest_call_hash]

    def _collect_work_finish_status(self) -> dict:
        work_finished_status = {work.name: self._has_work_finished(work) for work in self.works}
        assert len(work_finished_status) == len(self.works)
        return work_finished_status

    def _should_snapshot(self) -> bool:
        if len(self.works) == 0:
            return True
        elif self._has_updated:
            work_finished_status = self._collect_work_finish_status()
            if work_finished_status:
                return all(work_finished_status.values())
            else:
                return True
        return False

    def state_dict(self) -> Dict:
        return self.state

    def load_state_dict(self, state: Dict) -> None:
        self.set_state(state)

    def load_state_dict_from_checkpoint_dir(
        self,
        checkpoints_dir: str,
        version: Optional[int] = None,
    ) -> None:
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError(f"The provided directory `{checkpoints_dir}` doesn't exist.")
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith("v_") and f.endswith(".json")]
        if not checkpoints:
            raise Exception(f"No checkpoints where found in `{checkpoints_dir}`.")

        if version is None:
            # take the latest checkpoint.
            version = sorted(int(c.split("_")[1]) for c in checkpoints)[-1]

        available_checkpoints = [c for c in checkpoints if c.startswith(f"v_{version}_")]
        if not available_checkpoints:
            raise FileNotFoundError(f"The version `{version}` wasn't found in {checkpoints}.")
        elif len(available_checkpoints) > 1:
            raise Exception(f"Found 2 checkpoints `{available_checkpoints}`with the same version.")
        checkpoint_path = os.path.join(checkpoints_dir, available_checkpoints[0])
        state = pickle.load(open(checkpoint_path, "rb"))
        self.load_state_dict(state)

    def _dump_checkpoint(self) -> Optional[str]:
        checkpoints_dir = self.checkpoint_dir
        # TODO: Add supports to remotely saving checkpoints.
        if checkpoints_dir.startswith("s3:"):
            return None
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Get all current version within the provided folder and sort them
        checkpoint_versions = sorted(
            int(f.split("_")[1]) for f in os.listdir(checkpoints_dir) if f.startswith("v_") and f.endswith(".json")
        )

        if checkpoint_versions:
            previous_version = checkpoint_versions[-1]
        else:
            # initialization
            previous_version = -1

        checkpoint_path = os.path.join(checkpoints_dir, f"v_{previous_version + 1}_{time()}.json")

        with open(checkpoint_path, "wb") as f:
            pickle.dump(self.state_dict(), f)
        return checkpoint_path

    def connect(self, runtime: "Runtime") -> None:
        """Override to customize your application to the runtime."""
        pass

    def _on_cache_miss_exception(self) -> None:
        if self._has_updated:
            self._update_layout()

    def _register_schedule(self, schedule_hash: str, schedule_metadata: Dict) -> None:
        # create a thread only if a user uses the flow's schedule method.
        if not self._schedules:
            scheduler_thread = SchedulerThread(self)
            scheduler_thread.setDaemon(True)
            self.threads.append(scheduler_thread)
            self.threads[-1].start()
        self._schedules[schedule_hash] = deepcopy(schedule_metadata)

    def on_run_once_end(self) -> None:
        if not self._schedules:
            return
        # disable any flow schedules.
        for flow in self.flows:
            flow._disable_running_schedules()

    def _on_run_end(self):
        if os.getenv("LIGHTNING_DEBUG") == "2":
            del os.environ["LIGHTNING_DEBUG"]
            _console.setLevel(logging.INFO)

    @staticmethod
    def _extract_vars_from_component_name(component_name: str, state):
        child = state
        for child_name in component_name.split(".")[1:]:
            if child_name in child["flows"]:
                child = child["flows"][child_name]
            elif "structures" in child and child_name in child["structures"]:
                child = child["structures"][child_name]
            elif child_name in child["works"]:
                child = child["works"][child_name]
            else:
                return None

        # Filter private keys and drives
        return {
            k: v
            for k, v in child["vars"].items()
            if (
                not k.startswith("_")
                and not (isinstance(v, dict) and v.get("type", None) == "__drive__")
                and not (isinstance(v, (Payload, Path)))
            )
        }

    def _send_flow_to_work_deltas(self, state) -> None:
        if not self.flow_to_work_delta_queues:
            return

        for w in self.works:
            if not w.has_started:
                continue

            # Don't send changes when the state has been just sent.
            if w.run.has_sent:
                continue

            state_work = self._extract_vars_from_component_name(w.name, state)
            last_state_work = self._extract_vars_from_component_name(w.name, self._last_state)

            # Note: The work was dynamically created or deleted.
            if state_work is None or last_state_work is None:
                continue

            deep_diff = DeepDiff(last_state_work, state_work, verbose_level=2).to_dict()

            if "unprocessed" in deep_diff:
                deep_diff.pop("unprocessed")

            if deep_diff:
                logger.debug(f"Sending deep_diff to {w.name} : {deep_diff}")
                self.flow_to_work_delta_queues[w.name].put(deep_diff)

    def _start_with_flow_works(self):
        for w in self.works:
            if w._start_with_flow:
                parallel = w.parallel
                w._parallel = True
                w.start()
                w._parallel = parallel
