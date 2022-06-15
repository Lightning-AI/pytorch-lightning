import logging
import os
import pickle
import queue
import threading
import typing as t
import warnings
from copy import deepcopy
from time import time

from deepdiff import DeepDiff, Delta

import lightning_app
from lightning_app.core.constants import FLOW_DURATION_SAMPLES, FLOW_DURATION_THRESHOLD, STATE_ACCUMULATE_WAIT
from lightning_app.core.queues import BaseQueue, SingleProcessQueue
from lightning_app.frontend import Frontend
from lightning_app.storage.path import storage_root_dir
from lightning_app.utilities.app_helpers import _delta_to_appstate_delta, _LightningAppRef
from lightning_app.utilities.component import _convert_paths_after_init
from lightning_app.utilities.enum import AppStage
from lightning_app.utilities.exceptions import CacheMissException, ExitAppException
from lightning_app.utilities.layout import _collect_layout
from lightning_app.utilities.proxies import ComponentDelta
from lightning_app.utilities.scheduler import SchedulerThread
from lightning_app.utilities.tree import breadth_first
from lightning_app.utilities.warnings import LightningFlowWarning

if t.TYPE_CHECKING:
    from lightning_app.runners.backends.backend import Backend, WorkManager

logger = logging.getLogger(__name__)


class LightningApp:
    def __init__(
        self,
        root: "lightning_app.LightningFlow",
        debug: bool = False,
    ):
        """LightningApp, or App in short, alternatively run its root
        :class:`~lightning_app.core.flow.LightningFlow` component and collects state changes from external
        sources to maintain the application state up-to-date or performs checkpointing. All those operations
        are executed within an infinite loop.

        Arguments:
            root: The root LightningFlow component, that defined all the app's nested components, running infinitely.
            debug: Whether to run the application in debug model.

        .. doctest::

            >>> from lightning_app import LightningFlow, LightningApp
            >>> from lightning_app.runners import SingleProcessRuntime
            >>> class RootFlow(LightningFlow):
            ...     def run(self):
            ...         print("Hello World!")
            ...         self._exit()
            >>> app = LightningApp(RootFlow()) # application can be dispatched using the `runners`.
            >>> SingleProcessRuntime(app).dispatch()
            Hello World!
        """

        self._root = root

        # queues definition.
        self.delta_queue: t.Optional[BaseQueue] = None
        self.readiness_queue: t.Optional[BaseQueue] = None
        self.api_publish_state_queue: t.Optional[BaseQueue] = None
        self.api_delta_queue: t.Optional[BaseQueue] = None
        self.error_queue: t.Optional[BaseQueue] = None
        self.request_queues: t.Optional[t.Dict[str, BaseQueue]] = None
        self.response_queues: t.Optional[t.Dict[str, BaseQueue]] = None
        self.copy_request_queues: t.Optional[t.Dict[str, BaseQueue]] = None
        self.copy_response_queues: t.Optional[t.Dict[str, BaseQueue]] = None
        self.caller_queues: t.Optional[t.Dict[str, BaseQueue]] = None
        self.work_queues: t.Optional[t.Dict[str, BaseQueue]] = None

        self.should_publish_changes_to_api = False
        self.component_affiliation = None
        self.backend: t.Optional[Backend] = None
        _LightningAppRef.connect(self)
        self.processes: t.Dict[str, WorkManager] = {}
        self.frontends: t.Dict[str, Frontend] = {}
        self.stage = AppStage.RUNNING
        self._has_updated: bool = False
        self._schedules: t.Dict[str, t.Dict] = {}
        self.threads: t.List[threading.Thread] = []

        # NOTE: Checkpointing is disabled by default for the time being.  We
        # will enable it when resuming from full checkpoint is supported. Also,
        # we will need to revisit the logic at _should_snapshot, since right now
        # we are writing checkpoints too often, and this is expensive.
        self.checkpointing: bool = False

        self._update_layout()

        self._original_state = None
        self._last_state = self.state
        self.state_accumulate_wait = STATE_ACCUMULATE_WAIT

        self._last_run_time = 0.0
        self._run_times = []

        # Path attributes can't get properly attached during the initialization, because the full name
        # is only available after all Flows and Works have been instantiated.
        _convert_paths_after_init(self.root)

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

    def get_component_by_name(self, component_name: str):
        """Returns the instance corresponding to the given component name."""
        from lightning_app.structures import Dict, List
        from lightning_app.utilities.types import ComponentTuple

        if component_name == "root":
            return self.root
        if not component_name.startswith("root."):
            raise ValueError(f"Invalid component name {component_name}. Name must start with 'root'")

        current = self.root
        for child_name in component_name.split(".")[1:]:
            if isinstance(current, (Dict, List)):
                child = current[child_name] if isinstance(current, Dict) else current[int(child_name)]
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
        return os.path.join(storage_root_dir(), "checkpoints")

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
        diff = DeepDiff(last_state, new_state, view="tree")

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
    def get_state_changed_from_queue(q: BaseQueue, timeout: t.Optional[int] = None):
        try:
            delta = q.get(timeout=timeout or q.default_timeout)
            return delta
        except queue.Empty:
            return None

    def check_error_queue(self) -> None:
        exception: Exception = self.get_state_changed_from_queue(self.error_queue)
        if isinstance(exception, Exception):
            self.stage = AppStage.FAILED

    @property
    def flows(self) -> t.List["lightning_app.LightningFlow"]:
        """Returns all the flows defined within this application."""
        return [self.root] + self.root.get_all_children()

    @property
    def works(self) -> t.List["lightning_app.LightningWork"]:
        """Returns all the works defined within this application."""
        return self.root.works(recurse=True)

    @property
    def named_works(self) -> t.List[t.Tuple[str, "lightning_app.LightningWork"]]:
        """Returns all the works defined within this application with their names."""
        return self.root.named_works(recurse=True)

    def _collect_deltas_from_ui_and_work_queues(self) -> t.List[Delta]:
        # The aggregation would try to get as many deltas as possible
        # from both the `api_delta_queue` and `delta_queue`
        # during the `state_accumulate_wait` time.
        # The while loop can exit sooner if both queues are empty.

        should_get_delta_from_api = True
        should_get_component_output = True
        deltas = []
        t0 = time()

        while (time() - t0) < self.state_accumulate_wait:

            if self.api_delta_queue and should_get_delta_from_api:
                delta_from_api: Delta = self.get_state_changed_from_queue(self.api_delta_queue)  # TODO: rename
                if delta_from_api:
                    deltas.append(delta_from_api)
                else:
                    should_get_delta_from_api = False

            if self.delta_queue and should_get_component_output:
                component_output: t.Optional[ComponentDelta] = self.get_state_changed_from_queue(self.delta_queue)
                if component_output:
                    logger.debug(f"Received from {component_output.id} : {component_output.delta.to_dict()}")
                    work = self.get_component_by_name(component_output.id)
                    new_work_delta = _delta_to_appstate_delta(self.root, work, deepcopy(component_output.delta))
                    deltas.append(new_work_delta)
                else:
                    should_get_component_output = False

            # if both queues were found empties, should break the while loop.
            if not should_get_delta_from_api and not should_get_component_output:
                break

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

    def maybe_apply_changes(self) -> bool:
        """Get the deltas from both the flow queue and the work queue, merge the two deltas and update the state."""

        deltas = self._collect_deltas_from_ui_and_work_queues()

        if not deltas:
            # When no deltas are received from the Rest API or work queues,
            # we need to check if the flow modified the state and populate changes.
            if Delta(DeepDiff(self.last_state, self.state)).to_dict():
                # new_state = self.populate_changes(self.last_state, self.state)
                self.set_state(self.state)
                self._has_updated = True
            return False

        logger.debug(f"Received {[d.to_dict() for d in deltas]}")

        state = self.state
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
        self._has_updated = False
        self._last_run_time = 0.0

        if self.backend is not None:
            self.backend.update_work_statuses(self.works)

        self._update_layout()
        self.maybe_apply_changes()

        if self.checkpointing and self._should_snapshot():
            self._dump_checkpoint()

        if self.stage == AppStage.BLOCKING:
            return done

        if self.stage in (AppStage.STOPPING, AppStage.FAILED):
            return True

        elif self.stage == AppStage.RESTARTING:
            return self._apply_restarting()

        try:
            self.check_error_queue()
            t0 = time()
            self.root.run()
            self._last_run_time = time() - t0
        except CacheMissException:
            self._on_cache_miss_exception()
        except (ExitAppException, KeyboardInterrupt):
            done = True
            self.stage = AppStage.STOPPING

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

        if self.should_publish_changes_to_api and self.api_publish_state_queue:
            logger.debug("Publishing the state with changes")
            # Push two states to optimize start in the cloud.
            self.api_publish_state_queue.put(self.state)
            self.api_publish_state_queue.put(self.state)

        self._reset_run_time_monitor()

        while not done:
            done = self.run_once()

            self._update_run_time_monitor()

            if self._has_updated and self.should_publish_changes_to_api and self.api_publish_state_queue:
                self.api_publish_state_queue.put(self.state)

        return True

    def _update_layout(self) -> None:
        if self.backend:
            self.backend.resolve_url(self, base_url=None)

        for component in breadth_first(self.root, types=(lightning_app.LightningFlow,)):
            layout = _collect_layout(self, component)
            component._layout = layout

    def _apply_restarting(self) -> bool:
        self._reset_original_state()
        # apply stage after restoring the original state.
        self.stage = AppStage.BLOCKING
        return False

    def _collect_work_finish_status(self) -> dict:
        work_finished_status = {}
        for work in self.works:
            work_finished_status[work.name] = False
            for key in work._calls:
                if key == "latest_call_hash":
                    continue
                fn_metadata = work._calls[key]
                work_finished_status[work.name] = fn_metadata["name"] == "run" and "ret" in fn_metadata

        assert len(work_finished_status) == len(self.works)
        return work_finished_status

    def _should_snapshot(self) -> bool:
        if len(self.works) == 0:
            return True
        elif isinstance(self.delta_queue, SingleProcessQueue):
            return True
        elif self._has_updated:
            work_finished_status = self._collect_work_finish_status()
            if work_finished_status:
                return all(work_finished_status.values())
            else:
                return True
        return False

    def state_dict(self) -> t.Dict:
        return self.state

    def load_state_dict(self, state: t.Dict) -> None:
        self.set_state(state)

    def load_state_dict_from_checkpoint_dir(
        self,
        checkpoints_dir: str,
        version: t.Optional[int] = None,
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

    def _dump_checkpoint(self) -> t.Optional[str]:
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

    def connect(self, runtime: "lightning_app.runners.runtime.Runtime") -> None:
        """Override to customize your application to the runtime."""
        pass

    def _on_cache_miss_exception(self) -> None:
        if self._has_updated:
            self._update_layout()

    def _register_schedule(self, schedule_hash: str, schedule_metadata: t.Dict) -> None:
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
