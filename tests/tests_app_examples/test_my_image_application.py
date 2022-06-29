import os

from tests_app import _PROJECT_ROOT

from lightning_app.testing.helpers import RunIf
from lightning_app.testing.testing import application_testing, LightningTestApp
from lightning_app.utilities.state import AppState

SCRIPT_PATH = os.path.join(_PROJECT_ROOT, "examples/app_examples/image_application/app.py")


class ImageApplicationTestingTestApp(LightningTestApp):
    def download_data(self, state: AppState) -> None:
        state.downloader.should_download = True

    def start_training(self, state: AppState):
        state.image_classifier_train.train_work.selected_backbone = "resnet18"
        state.image_classifier_train.train_work.max_epochs = 1
        state.image_classifier_train.train_work.limit_train_batches = 1
        state.image_classifier_train.should_start = True

    def on_before_run_once(self):
        from PIL import Image

        from examples.image_application.demo_ui.demo import make_request

        if not self.root.downloader.has_completed:
            self.make_request(self.download_data)

        if not self.root.image_classifier_train.should_start:
            self.make_request(self.start_training)

        destination_dir = self.root.demo.destination_dir
        if destination_dir:
            predict_data = os.path.join(destination_dir, "predict")
            image_path = os.path.join(predict_data, os.listdir(predict_data)[0])
            response = make_request(Image.open(image_path), session=self._configure_session())
            assert "response" in response
            assert "request_time" in response
            return True


@RunIf(flash=True)
def test_image_application_example():
    """This test ensures image application example works properly."""

    command_line = [
        SCRIPT_PATH,
        "--blocking",
        "False",
        "--multiprocess",
        "--open-ui",
        "False",
    ]
    application_testing(ImageApplicationTestingTestApp, command_line)
