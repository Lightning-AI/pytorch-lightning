import os
from typing import Dict, List, Optional, Union

from core.components import TensorBoard, WeightsAndBiases
from core.components.script_runner import ScriptRunner

from lightning_app import LightningApp, LightningFlow
from lightning_app.frontend import StaticWebFrontend
from lightning_app.storage import Path
from lightning_app.utilities.packaging.cloud_compute import CloudCompute


class ReactUI(LightningFlow):
    def configure_layout(self):
        return StaticWebFrontend(str(Path(__file__).parent / "ui/build"))


class ScriptOrchestrator(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.script_runner: Optional[ScriptRunner] = None
        self.triggered: bool = False
        self.running: bool = False
        self.succeeded: bool = False
        self.failed: bool = False
        self.script_args: List[str] = []
        self.cloud_compute_args: Dict[str, Union[str, int]] = {"name": "cpu-small"}
        self.environment_variables: Dict[str, str] = {}
        self.script_path = "{{ script_path }}"

    def run(self) -> None:
        if not self.triggered:
            return

        if self.script_runner is None:
            self.script_runner = ScriptRunner(
                root_path=str(Path(__file__).parent / "source"),
                script_path=str(Path(__file__).parent / "source" / self.script_path),
                script_args=self.script_args,
                env=self._prepare_environment(),
                parallel=True,
                cloud_compute=CloudCompute(**self.cloud_compute_args),
                raise_exception=False,
            )
            self.script_runner.run()

        self.running = self.script_runner is not None and self.script_runner.has_started
        self.succeeded = self.script_runner is not None and self.script_runner.has_succeeded
        self.failed = self.script_runner is not None and self.script_runner.has_failed

        if self.succeeded or self.failed:
            self.triggered = False
            # TODO: support restarting
            # self.script_runner = None

    def _prepare_environment(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.environment_variables)
        return env


class Main(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.react_ui = ReactUI()
        self.script_orchestrator = ScriptOrchestrator()
        self.running_in_cloud = bool(os.environ.get("LIGHTNING_CLOUD_APP_ID", False))

    def run(self) -> None:
        self.react_ui.run()
        self.script_orchestrator.run()

        if self.script_orchestrator.script_runner and self.script_orchestrator.script_runner.logger_metadatas:
            if not getattr(self, "logger_component", None):
                # TODO: Hack with hasattr and setattr until
                #   https://linear.app/gridai/issue/LAI2-8970/work-getting-set-to-none-in-state-update-from-appstate
                #   is resolved
                logger_component = self._choose_logger_component()
                if logger_component is not None:
                    setattr(self, "logger_component", logger_component)
            else:
                self.logger_component.run()

    def configure_layout(self):
        tabs = [{"name": "Home", "content": self.react_ui}]
        if hasattr(self, "logger_component"):
            tabs.extend(self.logger_component.configure_layout())
        return tabs

    def _choose_logger_component(self) -> Optional[Union[TensorBoard, WeightsAndBiases]]:
        logger_metadatas = self.script_orchestrator.script_runner.logger_metadatas
        if not logger_metadatas:
            return
        if logger_metadatas[0].get("class_name") == "TensorBoardLogger":
            return TensorBoard(log_dir=self.script_orchestrator.script_runner.log_dir)
        if logger_metadatas[0].get("class_name") == "WandbLogger":
            return WeightsAndBiases(
                username=logger_metadatas[0]["username"],
                project_name=logger_metadatas[0]["project_name"],
                run_id=logger_metadatas[0]["run_id"],
                api_key=self.script_orchestrator.environment_variables.get("WANDB_API_KEY"),
            )


app = LightningApp(Main())
