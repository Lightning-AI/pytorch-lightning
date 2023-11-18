import os
import tempfile
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from lightning.app import CloudCompute
from lightning.app.components import TracerPythonScript
from optuna.distributions import CategoricalDistribution, LogUniformDistribution
from torchmetrics import Accuracy


class ObjectiveWork(TracerPythonScript):
    def __init__(self, script_path: str, data_dir: str, cloud_compute: Optional[CloudCompute]):
        timestamp = datetime.now().strftime("%H:%M:%S")
        tmpdir = tempfile.TemporaryDirectory().name
        submission_path = os.path.join(tmpdir, f"{timestamp}.csv")
        best_model_path = os.path.join(tmpdir, f"{timestamp}.model.pt")
        super().__init__(
            script_path,
            script_args=[
                f"--train_data_path={data_dir}/train",
                f"--test_data_path={data_dir}/test",
                f"--submission_path={submission_path}",
                f"--best_model_path={best_model_path}",
            ],
            cloud_compute=cloud_compute,
        )
        self.data_dir = data_dir
        self.best_model_path = best_model_path
        self.submission_path = submission_path
        self.metric = None
        self.trial_id = None
        self.metric = None
        self.params = None
        self.has_told_study = False

    def run(self, trial_id: int, **params):
        self.trial_id = trial_id
        self.params = params
        self.script_args.extend([f"--{k}={v}" for k, v in params.items()])
        super().run()
        self.compute_metric()

    def _to_labels(self, path: str):
        return torch.from_numpy(pd.read_csv(path).label.values)

    def compute_metric(self):
        self.metric = -1 * float(
            Accuracy(task="binary")(
                self._to_labels(self.submission_path),
                self._to_labels(f"{self.data_dir}/ground_truth.csv"),
            )
        )

    @staticmethod
    def distributions():
        return {
            "backbone": CategoricalDistribution(["resnet18", "resnet34"]),
            "learning_rate": LogUniformDistribution(0.0001, 0.1),
        }
