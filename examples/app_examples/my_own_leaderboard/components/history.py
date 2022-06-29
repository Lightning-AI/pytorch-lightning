import lightning as L
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState


class History(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.best_metrics = []
        self.timestamps = []

    def run(self, best_metric: float, timestamp: str):
        is_not_in = best_metric not in self.best_metrics
        if is_not_in:
            self.best_metrics.append(best_metric)
            self.timestamps.append(timestamp)

    def configure_layout(self):
        return StreamlitFrontend(history_render_fn)


def history_render_fn(state: AppState):
    import pandas as pd
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=2000, limit=None, key="refresh")

    st.title("LeaderBoard History")

    if state.best_metrics:
        df = pd.DataFrame({"date": state.timestamps, "metrics": state.best_metrics})
        df = df.set_index("date")
        st.line_chart(df)

    else:
        st.write("Launch a job from http://127.0.0.1:7501/view/Leaderboard")
