from pathlib import Path

import optuna
from hyperplot import HiPlotFlow
from lightning.app import CloudCompute, LightningApp, LightningFlow
from lightning.app.structures import Dict
from objective import ObjectiveWork


class RootHPOFlow(LightningFlow):
    def __init__(self, script_path, data_dir, total_trials, simultaneous_trials):
        super().__init__()
        self.script_path = script_path
        self.data_dir = data_dir
        self.total_trials = total_trials
        self.simultaneous_trials = simultaneous_trials
        self.num_trials = simultaneous_trials
        self._study = optuna.create_study()
        self.ws = Dict()
        self.hi_plot = HiPlotFlow()

    def run(self):
        if self.num_trials >= self.total_trials:
            self.stop()

        has_told_study = []

        for trial_idx in range(self.num_trials):
            work_name = f"objective_work_{trial_idx}"
            if work_name not in self.ws:
                objective_work = ObjectiveWork(
                    script_path=self.script_path,
                    data_dir=self.data_dir,
                    cloud_compute=CloudCompute("cpu"),
                )
                self.ws[work_name] = objective_work
            if not self.ws[work_name].has_started:
                trial = self._study.ask(ObjectiveWork.distributions())
                self.ws[work_name].run(trial_id=trial._trial_id, **trial.params)

            if self.ws[work_name].metric and not self.ws[work_name].has_told_study:
                self.hi_plot.data.append({"x": -1 * self.ws[work_name].metric, **self.ws[work_name].params})
                self._study.tell(self.ws[work_name].trial_id, self.ws[work_name].metric)
                self.ws[work_name].has_told_study = True

            has_told_study.append(self.ws[work_name].has_told_study)

        if all(has_told_study):
            self.num_trials += self.simultaneous_trials


if __name__ == "__main__":
    app = LightningApp(
        RootHPOFlow(
            script_path=str(Path(__file__).parent / "pl_script.py"),
            data_dir="data/hymenoptera_data_version_0",
            total_trials=6,
            simultaneous_trials=2,
        )
    )
