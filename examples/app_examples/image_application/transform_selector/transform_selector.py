import os

import torchvision.transforms as T
from PIL import Image

import lightning as L
from examples.image_application.shared import TRANSFORMS
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState


class TransformSelector(L.LightningFlow):
    def __init__(self):
        super().__init__()
        # `TransformSelector` internal state.
        self.selected_transform = "simple"
        self.destination_dir = None

    def run(self, destination_dir):
        self.destination_dir = destination_dir

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


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

    state.selected_transform = st.selectbox("Select your transform", TRANSFORMS.keys())

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(choice)
        st.image(image)

    with col2:
        transformed_image = TRANSFORMS[state.selected_transform](image)
        st.image(T.ToPILImage()(transformed_image))
