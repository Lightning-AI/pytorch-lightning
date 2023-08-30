from lightning.app import LightningFlow
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState


class HiPlotFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.data = []

    def run(self):
        pass

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state: AppState):
    import json

    import hiplot as hip
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st.set_page_config(layout="wide")
    st_autorefresh(interval=1000, limit=None, key="refresh")

    if not state.data:
        st.write("No data available yet ! Stay tuned")
        return

    xp = hip.Experiment.from_iterable(state.data)
    ret_val = xp.to_streamlit(ret="selected_uids", key="hip").display()
    st.markdown("hiplot returned " + json.dumps(ret_val))
