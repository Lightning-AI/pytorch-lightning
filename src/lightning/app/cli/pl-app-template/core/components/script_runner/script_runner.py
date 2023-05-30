import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from lightning.app.components.python import TracerPythonScript
from lightning.app.storage import Path
from lightning.app.utilities.packaging.build_config import BuildConfig, load_requirements
from lightning.app.utilities.tracer import Tracer


class ScriptRunner(TracerPythonScript):
    """The ScriptRunner executes the script using ``runpy`` and also patches the Trainer methods to inject
    additional code."""

    def __init__(self, root_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, cloud_build_config=self._get_build_config(root_path), **kwargs)
        self.root_path = root_path
        self.exception_message: str = ""
        self.trainer_progress: dict = {}
        self.trainer_state: dict = {}
        self.trainer_hparams: dict = {}
        self.model_hparams: dict = {}
        self.log_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None
        self.logger_metadatas: List[Dict[str, str]] = []

    def configure_tracer(self) -> Tracer:
        from core.callbacks import PLAppArtifactsTracker, PLAppProgressTracker, PLAppSummary, PLAppTrainerStateTracker

        from lightning.pytorch import Trainer

        tracer = Tracer()
        trainer_artifacts_tracker = PLAppArtifactsTracker(work=self)
        trainer_state_tracker = PLAppTrainerStateTracker(work=self)
        progress_tracker = PLAppProgressTracker(work=self)
        summary = PLAppSummary(work=self)

        def pre_trainer_init(_, *args: Any, **kwargs: Any) -> Tuple[Dict, Tuple[Any, ...], Dict[str, Any]]:
            kwargs.setdefault("callbacks", [])
            kwargs["callbacks"].extend(
                [
                    trainer_artifacts_tracker,
                    trainer_state_tracker,
                    progress_tracker,
                    summary,
                ]
            )
            return {}, args, kwargs

        tracer.add_traced(Trainer, "__init__", pre_fn=pre_trainer_init)
        return tracer

    def run(self) -> None:
        self.exception_message = ""
        # We need to set the module path both in sys.path and the PYTHONPATH env variable.
        # The former is for the current process which is already running, and the env variable is needed in case
        # the script launches subprocesses
        sys.path.insert(0, self.root_path)
        self.env["PYTHONPATH"] = self.root_path
        super().run()

    def on_exception(self, exception: BaseException) -> None:
        self.exception_message = traceback.format_exc()
        super().on_exception(exception)

    @staticmethod
    def _get_build_config(root_path: str) -> Optional[BuildConfig]:
        # These are the requirements for the script runner itself
        requirements = [
            "protobuf<4.21.0",
            "pytorch-lightning<=1.6.3",
            "pydantic<=1.9.0",
        ]
        if Path(root_path, "requirements.txt").exists():
            # Requirements from the user's code folder
            requirements.extend(load_requirements(root_path, file_name="requirements.txt"))

        return BuildConfig(requirements=requirements)
