import base64
import os
from io import BytesIO
from time import time

import requests
from PIL import Image

import lightning as L
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState


class DemoUI(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.destination_dir = None
        self.requests_count = 0

    def run(self, destination_dir: str):
        self.destination_dir = destination_dir

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def make_request(image, session=None):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("UTF-8")
    body = {"session": "UUID", "payload": {"inputs": {"data": img_str}}}
    t0 = time()
    client = session if session else requests
    resp = client.post("http://127.0.0.1:8000/predict", json=body)
    t1 = time()
    return {"response": resp.json(), "request_time": t1 - t0}


def _render_streamlit_fn(state: AppState):
    """This method would be running StreamLit within its own process.

    Arguments:
        state: Connection to this flow state. Enable changing the state from the UI.
    """
    import streamlit as st

    images = []

    if not isinstance(state.destination_dir, str):
        st.write("Please, go to the Data Download Tab to download the data.")
        return

    for root, _, files in os.walk(state.destination_dir):
        for f in files:
            if "jpg" in f and f not in images:
                images.append(os.path.join(root, f))

    if not images:
        st.write("No images were found.")
        return

    choice = st.selectbox("Select an Image", images)

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(choice)
        st.image(image)

    with col2:
        state.requests_count = state.requests_count + 1
        st.write(f"Request made {state.requests_count}")
        st.write("Received Prediction")
        st.json(make_request(image))
