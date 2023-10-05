import os

from lightning.app import LightningFlow, LightningApp
from lightning.app.frontend import StaticWebFrontend, StreamlitFrontend
from lightning.app.utilities.state import AppState


# Step 1: Define your LightningFlow component with the app UI
class UIStreamLit(LightningFlow):
    def __init__(self):
        super().__init__()
        self.should_print = False

    # Step 2: Override `configure_layout` to define the layout of the UI
    # In this example, we are using `StreamlitFrontend`
    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


# Step 3: Implement the StreamLit render method
def render_fn(state: AppState):
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=2000, limit=None, key="refresh")

    state.should_print = st.select_slider(
        "Should the Application print 'Hello World !' to the terminal:",
        [False, True],
    )


# Step 4: Implement a Static Web Frontend. This could be react, vue, etc.
class UIStatic(LightningFlow):
    # Step 5:
    def configure_layout(self):
        return StaticWebFrontend(os.path.join(os.path.dirname(__file__), "ui"))


# Step 6: Implement the root flow.
class HelloWorld(LightningFlow):
    def __init__(self):
        super().__init__()
        self.static_ui = UIStatic()
        self.streamlit_ui = UIStreamLit()

    def run(self):
        print("Hello World!" if self.streamlit_ui.should_print else "")

    def configure_layout(self):
        return [
            {"name": "StreamLit", "content": self.streamlit_ui},
            {"name": "Static", "content": self.static_ui},
        ]


app = LightningApp(HelloWorld())
