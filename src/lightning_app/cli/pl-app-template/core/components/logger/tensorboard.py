import subprocess
import time
from typing import Dict, List

from lightning.app import BuildConfig, LightningFlow, LightningWork
from lightning.app.storage import Path


class TensorBoard(LightningFlow):
    def __init__(self, log_dir: Path, sync_every_n_seconds: int = 5) -> None:
        """This TensorBoard component synchronizes the log directory of an experiment and starts up the server.

        Args:
            log_dir: The path to the directory where the TensorBoard log-files will appear.
            sync_every_n_seconds: How often to sync the log directory (given as an argument to the run method)
        """
        super().__init__()
        self.worker = TensorBoardWorker(log_dir=log_dir, sync_every_n_seconds=sync_every_n_seconds)

    def run(self) -> None:
        self.worker.run()

    def configure_layout(self) -> List[Dict[str, str]]:
        return [{"name": "Training Logs", "content": self.worker.url}]


class TensorBoardWorker(LightningWork):
    def __init__(self, log_dir: Path, sync_every_n_seconds: int = 5) -> None:
        super().__init__(cloud_build_config=BuildConfig(requirements=["tensorboard"]))
        self.log_dir = log_dir
        self._sync_every_n_seconds = sync_every_n_seconds

    def run(self) -> None:
        subprocess.Popen(
            [
                "tensorboard",
                "--logdir",
                str(self.log_dir),
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]
        )

        # Download the log directory periodically
        while True:
            time.sleep(self._sync_every_n_seconds)
            if self.log_dir.exists_remote():
                self.log_dir.get(overwrite=True)
