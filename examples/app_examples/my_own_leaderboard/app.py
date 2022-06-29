import os

import lightning as L
from examples.my_own_leaderboard.components.history import History
from examples.my_own_leaderboard.components.leaderboard import LeaderBoard
from examples.my_own_leaderboard.components.utils import download_data


class RootFlow(L.LightningFlow):
    def __init__(self, train_data_path: str, test_data_path: str, ground_truth_data_path: str):
        super().__init__()
        self.leaderboard = LeaderBoard(train_data_path, test_data_path, ground_truth_data_path)
        self.history = History()

    def run(self):
        self.leaderboard.run()
        if self.leaderboard.best_metric:
            self.history.run(self.leaderboard.best_metric, self.leaderboard.timestamp)

    def configure_layout(self):
        return [{"name": "Leaderboard", "content": self.leaderboard}, {"name": "History", "content": self.history}]


if __name__ == "__main__":

    # download some data for demonstration purpose
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
    download_data("https://pl-flash-data.s3.amazonaws.com/kaggle.zip", DATA_PATH)

    app = L.LightningApp(
        RootFlow(
            train_data_path=os.path.join(DATA_PATH, "kaggle", "train.csv"),
            test_data_path=os.path.join(DATA_PATH, "kaggle", "test.csv"),
            ground_truth_data_path=os.path.join(DATA_PATH, "kaggle", "test_ground_truth.csv"),
        )
    )
