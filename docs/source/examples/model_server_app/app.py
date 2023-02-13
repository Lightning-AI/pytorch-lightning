from locust_component import Locust
from model_server import MLServer
from train import TrainModel

from lightning import LightningApp, LightningFlow


class TrainAndServe(LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_model = TrainModel()
        self.model_server = MLServer(
            name="mnist-svm",
            implementation="mlserver_sklearn.SKLearnModel",
            workers=8,
        )
        self.performance_tester = Locust(num_users=100)

    def run(self):
        self.train_model.run()
        self.model_server.run(self.train_model.best_model_path)
        if self.model_server.alive():
            # The performance tester needs the model server to be up
            # and running to be started, so the URL is added in the UI.
            self.performance_tester.run(self.model_server.url)

    def configure_layout(self):
        return [
            {"name": "Server", "content": self.model_server.url + "/docs"},
            {"name": "Server Testing", "content": self.performance_tester},
        ]


app = LightningApp(TrainAndServe())
