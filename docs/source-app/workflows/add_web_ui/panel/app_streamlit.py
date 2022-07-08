# app.py
import os
from typing import Union

import streamlit as st

import lightning as L
from lightning.app.frontend.stream_lit import StreamlitFrontend
from lightning_app.core.flow import LightningFlow
from lightning_app.utilities.state import AppState


def your_streamlit_app(lightning_app_state):
    st.write("hello world")
    st.write(lightning_app_state)


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
