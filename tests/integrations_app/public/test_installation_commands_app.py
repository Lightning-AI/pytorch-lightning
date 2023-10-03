import os

import pytest
from lightning.app.testing.testing import run_app_in_cloud

from integrations_app.public import _PATH_EXAMPLES


@pytest.mark.cloud()
def test_installation_commands_app_example_cloud() -> None:
    # This is expected to pass, since the "setup" flag is passed
    with run_app_in_cloud(
        os.path.join(_PATH_EXAMPLES, "installation_commands"),
        app_name="app.py",
        extra_args=["--setup"],
        debug=True,
    ) as (_, _, fetch_logs, _):
        has_logs = False
        while not has_logs:
            for log in fetch_logs(["work"]):
                if "lmdb successfully installed" in log:
                    has_logs = True
