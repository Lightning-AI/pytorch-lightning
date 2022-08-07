import logging
import sys

import optuna
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def objective(trial):
    iris = datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - clf.score(valid_x, valid_y)


# Add stream handler of stdout to show the messages
logger = optuna.logging.get_logger("optuna")
logger.addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
