import argparse
import os
from typing import Dict, Optional

from flash import Trainer
from flash.core.data.transforms import ApplyToKeys
from flash.image import ImageClassificationData, ImageClassifier

import lightning as L
from examples.image_application.shared import TRANSFORMS
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase


class SimpleLogger(LightningLoggerBase):
    def __init__(self, work: "L.LightningWork"):
        super().__init__()
        self.work = work

    @property
    def name(self) -> str:
        return "lightning_logs"

    @property
    def version(self) -> int:
        return 0

    def experiment(self):
        return None

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        self.work.params = params

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self.work.metrics = metrics


class SimpleTracker(Callback):
    def __init__(self, work):
        super().__init__()
        self.work = work

    def on_train_batch_end(self, trainer, *_, **__) -> None:
        self.work.global_step = trainer.global_step + 1


class ImageClassifierTrainWork(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_folder: Optional[str] = None
        self.selected_transform: Optional[str] = None
        self.best_model_path: Optional[str] = None
        self.max_epochs = 5
        self.limit_train_batches = 1
        self.limit_val_batches = 1
        self.selected_backbone = None
        self.params = {}
        self.global_step = 0
        self.metrics = {}

    def run(self, image_folder: str, selected_transform: str):
        self.image_folder = image_folder
        self.selected_transform = selected_transform

        transform = ApplyToKeys("input", TRANSFORMS[selected_transform])

        datamodule = ImageClassificationData.from_folders(
            train_folder=os.path.join(self.image_folder, "train"),
            val_folder=os.path.join(self.image_folder, "val"),
            test_folder=os.path.join(self.image_folder, "test"),
            train_transform=transform,
            val_transform=transform,
            test_transform=transform,
            predict_transform=transform,
            batch_size=8,
        )

        model = ImageClassifier(datamodule.num_classes, backbone=self.selected_backbone)

        trainer = Trainer(
            max_epochs=self.max_epochs,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            num_sanity_val_steps=0,
            logger=SimpleLogger(self),
            callbacks=[SimpleTracker(self)],
        )

        trainer.fit(model, datamodule=datamodule)

        # store the best model path within the `ImageClassifierTrainWork` state.
        self.best_model_path = trainer.checkpoint_callback.best_model_path


class ImageClassifierTrainFlow(L.LightningFlow):
    def __init__(self, should_start: bool = False):
        super().__init__()
        """
        The ImageClassifierTrainFlow is responsible to create an `ImageClassifierTrainWork`.

        Arguments:
            should_start: Whether training should start when the flow is running.
        """
        self.train_work = ImageClassifierTrainWork()
        self.should_start = should_start

    def run(self, destination_dir: str, selected_transform: str):
        if self.should_start:
            self.train_work.run(destination_dir, selected_transform)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState) -> None:
    """This method would be running StreamLit within its own process.

    Arguments:
        state: Connection to this flow state. Enable changing the state from the UI.
    """
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=2000, limit=None, key="refresh")

    work_state = state.train_work

    col1, col2, col3 = st.columns(3)
    with col1:
        work_state.max_epochs = int(st.number_input(value=work_state.max_epochs, label="Select `max_epochs`."))

    with col2:
        work_state.limit_train_batches = int(
            st.number_input(value=work_state.limit_train_batches, label="Select `limit_train_batches`.")
        )

    with col3:
        work_state.limit_val_batches = int(
            st.number_input(value=work_state.limit_val_batches, label="Select `limit_val_batches`.")
        )

    backbones = ImageClassifier.available_backbones()
    work_state.selected_backbone = st.selectbox(f"Select your backbone among {len(backbones)} options", backbones)

    should_start = st.button("Should start training ?")

    if should_start:
        state.should_start = should_start

    if state.should_start:

        total_training_steps = work_state.max_epochs * work_state.limit_train_batches
        st.progress(float(work_state.global_step) / total_training_steps)

        st.write(f"Best model path: {work_state.best_model_path}.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Model Parameters")
            st.json(work_state.params)

        with col2:
            st.write("Current Training Step")
            st.json({"step": work_state.global_step})

        with col3:
            st.write("Current Metrics")
            st.write(work_state.metrics)


# Used to test locally that the Flow works as expected.
if __name__ == "__main__":
    flow = ImageClassifierTrainFlow(should_start=True)
    flow.run("storage/1234/root/downloader/work/data/hymenoptera_data", "randaugment")
