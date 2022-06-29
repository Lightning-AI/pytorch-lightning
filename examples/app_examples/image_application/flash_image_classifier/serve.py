from flash.image import ImageClassifier

import lightning as L


class ImageClassifierServeWork(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, parallel=True)

    def run(self, best_model_path):
        model = ImageClassifier.load_from_checkpoint(best_model_path)
        model.serve()
