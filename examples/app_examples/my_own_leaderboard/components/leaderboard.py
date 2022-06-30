import os
from datetime import datetime
from subprocess import Popen
from typing import Optional

import lightning as L
from examples.my_own_leaderboard.components.validation_layer import ValidationLayer
from lightning.app.frontend import StreamlitFrontend
from lightning.app.structures import List
from lightning.app.utilities.state import AppState


class PythonScript(L.LightningWork):
    def __init__(
        self, script_path, train_data_path: str, test_data_path: str, cloud_compute: Optional[L.CloudCompute] = None
    ):
        super().__init__(cloud_compute=cloud_compute, parallel=True)
        self.script_path = script_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.submission_path = None
        self.has_completed = False

    def run(self):
        submission_name = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        submission_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submissions")
        if not os.path.exists(submission_dir):
            os.makedirs(submission_dir)
        self.submission_path = os.path.join(submission_dir, f"{submission_name}_submission.csv")
        proc = Popen(
            [
                "python",
                f"{self.script_path}",
                "--submission_path",
                self.submission_path,
                "--train_data_path",
                self.train_data_path,
                "--test_data_path",
                self.test_data_path,
            ]
        )
        proc.wait()
        self.has_completed = True


class LeaderBoard(L.LightningFlow):

    """The LeaderBoard would keep track of a queue of requested Work. Work gets requested by the user through the
    UI.

    Once a work is finished, the submission file is compared to the ground truth and the associated metric is computed.
    Its metadata moves from the Queue Work to completed Work.
    """

    def __init__(self, train_data_path: str, test_data_path: str, ground_truth_data_path: str):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.queue = []
        self.completed_works = []
        self.current_request = None
        self.work = None
        self.validation_layer = ValidationLayer(self.ground_truth_data_path)
        self.created_works = List()
        self.counter = 0

    @property
    def best_metric(self):
        return self.validation_layer.best_metric

    @property
    def timestamp(self):
        return self.validation_layer.timestamp

    def run(self):
        if self.queue and self.current_request is None:
            self.current_request = self.queue[0]
            self.created_works.append(
                PythonScript(
                    **self.current_request,
                    train_data_path=self.train_data_path,
                    test_data_path=self.test_data_path,
                )
            )
            self.created_works[-1].run()

        if self.current_request and self.created_works[-1].has_completed:
            request_id = len(self.completed_works)
            self.validation_layer.run(
                request_id,
                self.created_works[-1].submission_path,
            )
            self.current_request["id"] = request_id
            self.current_request["current_metric"] = self.validation_layer.current_metric
            self.current_request["timestamp"] = self.validation_layer.timestamp
            self.completed_works.append(self.current_request)
            self.completed_works = sorted(self.completed_works, key=lambda x: x["current_metric"])
            self.current_request = None
            self.queue.pop(0)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state: AppState):
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=1000, limit=None, key="refresh")

    st.title("My Own LeaderBoard")

    scripts_path = "examples/my_own_leaderboard/scripts"
    choices = [os.path.join(scripts_path, f) for f in os.listdir(scripts_path) if f.endswith(".py")]

    choice = st.selectbox("Select a Script", choices)
    should_run = st.button("Run the selected script ?")

    if should_run:
        state.queue = state.queue + [{"script_path": choice}]

    col1, col2 = st.columns(2)

    with col1:
        st.write("Queued Works")
        st.json(state.queue)

    with col2:
        st.write("Finished Works")
        st.json(state.completed_works)
