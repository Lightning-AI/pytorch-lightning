import optuna
from objective import ObjectiveWork
from optuna.distributions import CategoricalDistribution, LogUniformDistribution

TOTAL_TRIALS = 6
SIMULTANEOUS_TRIALS = 2
NUM_TRIALS = SIMULTANEOUS_TRIALS
DONE = False

STUDY = optuna.create_study()
DISTRIBUTIONS = {
    "backbone": CategoricalDistribution(["resnet18", "resnet34"]),
    "learning_rate": LogUniformDistribution(0.0001, 0.1),
}
TRIALS = [ObjectiveWork() for _ in range(TOTAL_TRIALS)]

# Lightning Infinite Loop
while not DONE:

    # Finish the Hyperparameter Optimization
    if NUM_TRIALS >= TOTAL_TRIALS:
        DONE = True
        continue

    has_told_study = []

    # Iterate over the possible number of trials.
    for trial_idx in range(NUM_TRIALS):

        objective_work = TRIALS[trial_idx]

        # If a work has already started, it won't be started again.
        if not objective_work.has_started:
            # Sample a new trial from the distributions
            trial = STUDY.ask(DISTRIBUTIONS)
            # Run the work
            objective_work.run(trial_id=trial._trial_id, **trial.params)

        # With Lightning, the `objective_work` will run asynchronously
        # and the metric will be prodcued after X amount of time.
        # The Lightning Infinite Loop would have run a very large number of times by then.
        if objective_work.metric and not objective_work.has_told_study:
            # Add the metric in the Study
            STUDY.tell(objective_work.trial_id, objective_work.metric)
            objective_work.has_told_study = True

        # Keep track if the objective work has populated the study.
        has_told_study.append(objective_work.has_told_study)

    # Trigger the next trials.
    if all(has_told_study):
        NUM_TRIALS += SIMULTANEOUS_TRIALS

print({w.trial_id: w.metric for w in TRIALS})
