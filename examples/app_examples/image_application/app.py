import lightning as L
from examples.image_application.demo_ui.demo import DemoUI
from examples.image_application.downloader.downloader import Downloader
from examples.image_application.flash_image_classifier.serve import ImageClassifierServeWork
from examples.image_application.flash_image_classifier.train import ImageClassifierTrainFlow
from examples.image_application.transform_selector.transform_selector import TransformSelector


class ImageExplorer(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.downloader = Downloader()
        self.transform_selector = TransformSelector()
        self.image_classifier_train = ImageClassifierTrainFlow()
        self.image_classifier_serve = ImageClassifierServeWork()
        self.demo = DemoUI()

    def run(self):
        self.downloader.run()
        if self.downloader.has_completed:
            self.transform_selector.run(self.downloader.destination_dir)
            if self.transform_selector.selected_transform:
                self.image_classifier_train.run(
                    self.downloader.destination_dir, self.transform_selector.selected_transform
                )
            best_model_path = self.image_classifier_train.train_work.best_model_path
            if best_model_path:
                self.demo.run(self.downloader.destination_dir)
                self.image_classifier_serve.run(best_model_path)

    def configure_layout(self):
        return [
            {"name": "Data-Downloader", "content": self.downloader},
            {"name": "Data-Augmentation-Selector", "content": self.transform_selector},
            {"name": "Flash-Image-Classifier-Training", "content": self.image_classifier_train},
            {"name": "Flash-Image-Classifier-Serving", "content": "http://127.0.0.1:8000/docs"},
            {"name": "Demo", "content": self.demo},
        ]


app = L.LightningApp(ImageExplorer())
