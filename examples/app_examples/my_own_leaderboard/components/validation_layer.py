from datetime import datetime

import torch
from torchmetrics import MeanAbsoluteError

import lightning as L


class ValidationLayer(L.LightningWork):
    def __init__(self, ground_truth: str):
        super().__init__(parallel=False, cache_calls=False)
        self.ground_truth = ground_truth
        self.best_metric = None
        self.timestamp = None
        self.work_id = None
        self.current_metric = None

    def run(self, work_id: int, submission_path: str):
        import pandas as pd

        ground_truth = torch.from_numpy(pd.read_csv(self.ground_truth)["num_sold"].values)
        submission = torch.from_numpy(pd.read_csv(submission_path)["num_sold"].values)
        metric = MeanAbsoluteError()
        self.current_metric = metric(submission, ground_truth).item()
        if self.best_metric is None:
            self.best_metric = self.current_metric
            self.work_id = work_id
            self.timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        elif self.current_metric < self.best_metric:
            self.best_metric = self.current_metric
            self.work_id = work_id
            self.timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


if __name__ == "__main__":
    validation_layer = ValidationLayer("examples/my_own_leaderboard/hidden_data/test_ground_truth.csv")
    validation_layer.run(0, "examples/my_own_leaderboard/scripts/submission.csv")
    assert validation_layer.work_id == 0
