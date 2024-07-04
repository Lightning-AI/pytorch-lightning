import os
from importlib import import_module

import pandas as pd
import torch
import torch.nn as nn
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.components import TracerPythonScript
from lightning.app.storage import Payload
from lightning.app.structures import Dict, List
from sklearn import datasets


def get_path(path):
    return os.path.join(os.path.dirname(__file__), path)


class GetDataWork(LightningWork):
    """This component is responsible to download some data and store them with a PayLoad."""

    def __init__(self):
        super().__init__()
        self.df_data = None
        self.df_target = None

    def run(self):
        print("Starting data collection...")
        data = datasets.fetch_california_housing(data_home=get_path("data"))
        self.df_data = Payload(pd.DataFrame(data["data"], columns=data["feature_names"]))
        self.df_target = Payload(pd.DataFrame(data["target"], columns=["MedHouseVal"]))
        print("Finished data collection.")


class ModelWork(LightningWork):
    """This component is receiving some data and train a sklearn model."""

    def __init__(self, model_path: str, parallel: bool):
        super().__init__(parallel=parallel)
        self.model_path, self.model_name = model_path.split(".")
        self.test_rmse = None

    def run(self, X_train: Payload, X_test: Payload, y_train: Payload, y_test: Payload):
        print(f"Starting training and evaluating {self.model_name}...")
        module = import_module(f"sklearn.{self.model_path}")
        model = getattr(module, self.model_name)()

        # Convert pandas DataFrames to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.value.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.value.values, dtype=torch.float32).reshape(-1)
        X_test_tensor = torch.tensor(X_test.value.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.value.values, dtype=torch.float32).reshape(-1)

        # Fit the model (still using numpy as sklearn expects numpy arrays)
        model.fit(X_train_tensor.numpy(), y_train_tensor.numpy())

        # Predict using PyTorch
        y_test_prediction = torch.tensor(model.predict(X_test_tensor.numpy()), dtype=torch.float32)

        # Calculate the MSE using PyTorch
        criterion = nn.MSELoss()
        mse = criterion(y_test_prediction, y_test_tensor)

        # Calculate the RMSE
        self.test_rmse = torch.sqrt(mse).item()

        print(f"Finished training and evaluating {self.model_name}.")


class DAG(LightningFlow):
    """This component is a DAG."""

    def __init__(self, models_paths: list):
        super().__init__()
        self.data_collector = GetDataWork()
        self.processing = TracerPythonScript(
            get_path("processing.py"),
            outputs=["X_train", "X_test", "y_train", "y_test"],
        )

        self.dict = Dict(**{
            model_path.split(".")[-1]: ModelWork(model_path, parallel=True) for model_path in models_paths
        })

        self.has_completed = False
        self.metrics = {}

    def run(self):
        self.data_collector.run()
        self.processing.run(
            df_data=self.data_collector.df_data,
            df_target=self.data_collector.df_target,
        )

        for model, work in self.dict.items():
            work.run(
                X_train=self.processing.X_train,
                X_test=self.processing.X_test,
                y_train=self.processing.y_train,
                y_test=self.processing.y_test,
            )
            if work.test_rmse:  # Use the state to control when to collect and stop.
                self.metrics[model] = work.test_rmse
                work.stop()  # Stop the model work to reduce cost

        # Step 4: Print the score of each model when they are all finished.
        if len(self.metrics) == len(self.dict):
            print(self.metrics)
            self.has_completed = True


class ScheduledDAG(LightningFlow):
    def __init__(self, dag_cls, **dag_kwargs):
        super().__init__()
        self.dags = List()
        self._dag_cls = dag_cls
        self.dag_kwargs = dag_kwargs

    def run(self):
        """Example of scheduling an infinite number of DAG runs continuously."""
        # Step 1: Every minute, create and launch a new DAG.
        if self.schedule("* * * * *"):
            print("Launching a new DAG")
            self.dags.append(self._dag_cls(**self.dag_kwargs))

        for dag in self.dags:
            if not dag.has_completed:
                dag.run()


app = LightningApp(
    ScheduledDAG(
        DAG,
        models_paths=[
            "svm.SVR",
            "linear_model.LinearRegression",
            "tree.DecisionTreeRegressor",
        ],
    ),
)
