import numpy as np

from lightning_app import LightningWork


class ObjectiveWork(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.has_started = False
        self.metric = None
        self.trial_id = None
        self.params = None
        self.has_told_study = False

    def run(self, trial_id, **params):
        self.trial_id = trial_id
        self.params = params  # Received suggested `backbone` and `learning_rate`
        self.has_started = True
        # Emulate metric computation would be computed once a script has been completed.
        # In reality, this would excute a user defined script.
        self.metric = np.random.uniform(0, 100)
