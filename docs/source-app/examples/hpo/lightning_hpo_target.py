import optuna
from lightning_hpo import BaseObjective, Optimizer
from optuna.distributions import LogUniformDistribution
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from lightning import LightningApp, LightningFlow


class Objective(BaseObjective):
    def run(self, params):
        # WARNING: Don't forget to assign `params` to self,
        # so they get tracked in the state.
        self.params = params

        iris = datasets.load_iris()
        classes = list(set(iris.target))
        train_x, valid_x, train_y, valid_y = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

        clf = SGDClassifier(alpha=params["alpha"])

        for step in range(100):
            clf.partial_fit(train_x, train_y, classes=classes)
            intermediate_value = 1.0 - clf.score(valid_x, valid_y)

            # WARNING: Assign to reports,
            # so the state is instantly sent to the flow.
            self.reports = self.reports + [[intermediate_value, step]]

        self.best_model_score = 1.0 - clf.score(valid_x, valid_y)

    def distributions(self):
        return {"alpha": LogUniformDistribution(1e-5, 1e-1)}


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.optimizer = Optimizer(
            objective_cls=Objective,
            n_trials=20,
            study=optuna.create_study(pruner=optuna.pruners.MedianPruner()),
        )

    def run(self):
        self.optimizer.run()

    def configure_layout(self):
        return {"name": "HyperPlot", "content": self.optimizer.hi_plot}


app = LightningApp(RootFlow())
