# app.py
from typing import Union
import lightning as L
from lightning.app.frontend.stream_lit import StreamlitFrontend
import streamlit as st
from lightning_app.core.flow import LightningFlow
from lightning_app.utilities.state import AppState
import os

def your_streamlit_app(lightning_app_state):
    st.write('hello world')
    st.write(lightning_app_state)
    st.write(os.environ["LIGHTNING_FLOW_NAME"])

class LitStreamlit(L.LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=your_streamlit_app)

class LitApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_streamlit = LitStreamlit()

    def configure_layout(self):
        tab1 = {"name": "home", "content": self.lit_streamlit}
        return tab1

app = L.LightningApp(LitApp())