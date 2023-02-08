import os
from time import sleep

from integrations_app.flagship import _PATH_INTEGRATIONS_DIR

from lightning.app.testing.testing import run_app_in_cloud


def test_app_in_cloud():

    with run_app_in_cloud(_PATH_INTEGRATIONS_DIR, "dummy_test_app.py") as (
        _,
        view_page,
        fetch_logs,
        name,
    ):

        # Validate the logs.
        has_logs = False
        while not has_logs:
            logs = list(fetch_logs)
            for log in logs:
                if "`Trainer.fit` stopped: `max_epochs=2` reached." in log:
                    has_logs = True
            sleep(1)

    expected_strings = [
        # don't include values for actual hardware availability as this may depend on environment.
        "GPU available: ",
        "All distributed processes registered.",
        "674 K    Trainable params\n0         Non - trainable params\n674 K    Total params\n2.699   Total estimated model params size(MB)",
        "Epoch 0:",
        "`Trainer.fit` stopped: `max_epochs=2` reached.",
        "Input text:Input text:\n summarize: ML Ops platforms come in many flavors from platforms that train models",
    ]

    for curr_str in expected_strings:
        assert any([curr_str in line for line in logs])
    os.remove(os.path.join(_PATH_INTEGRATIONS_DIR, "test_app.py"))
