import os

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import application_testing, LightningTestApp
from lightning_app.utilities.state import AppState


class LeaderBoardTestingTestApp(LightningTestApp):
    def trigger_queue(self, state: AppState):
        script_path = os.path.join(_PROJECT_ROOT, "examples/app_examples/my_own_leaderboard/scripts/random_script.py")
        state.leaderboard.queue = state.leaderboard.queue + [{"script_path": script_path}]

    def on_before_run_once(self):
        if self.counter == 1:
            self.make_request(self.trigger_queue)
        if self.root.leaderboard.validation_layer.best_metric:
            return True


@pytest.mark.skipif(True, reason="TODO: Resolve the flakyness of this test.")
def test_my_own_leaderboard_example():

    """This test ensures my own leaderboard example works properly."""

    command_line = [
        os.path.join(_PROJECT_ROOT, "examples/app_examples/my_own_leaderboard/app.py"),
        "--blocking",
        "False",
        "--multiprocess",
        "--open-ui",
        "False",
    ]
    application_testing(LeaderBoardTestingTestApp, command_line)
