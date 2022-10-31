import logging

from lightning_app import LightningApp, LightningFlow
from lightning_app.frontend import StreamlitFrontend
from lightning_app.utilities.state import AppState

logger = logging.getLogger(__name__)


class StreamlitUI(LightningFlow):
    def __init__(self):
        super().__init__()
        self.message_to_print = "Hello World!"
        self.should_print = False

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state: AppState):
    import streamlit as st

    should_print = st.button("Should print to the terminal ?")

    if should_print:
        state.should_print = not state.should_print

    st.write("Currently printing." if state.should_print else "Currently waiting to print.")


class HelloWorld(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.streamlit_ui = StreamlitUI()

    def run(self):
        self.streamlit_ui.run()
        if self.streamlit_ui.should_print:
            logger.info(f"{self.counter}: {self.streamlit_ui.message_to_print}")
            self.counter += 1
            self.streamlit_ui.should_print = False

    def configure_layout(self):
        return [{"name": "StreamLitUI", "content": self.streamlit_ui}]


app = LightningApp(HelloWorld(), debug=True)
