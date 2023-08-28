import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute
from lightning.pytorch.callbacks import ModelCheckpoint


# Step 1: Create a training LightningWork component that gets a backbone as input
# and saves the best model and its score
class ImageClassifierTrainWork(LightningWork):
    def __init__(self, max_epochs: int, backbone: str, cloud_compute: CloudCompute):
        # parallel is set to True to run asynchronously
        super().__init__(parallel=True, cloud_compute=cloud_compute)
        # Number of epochs to run
        self.max_epochs = max_epochs
        # The model backbone to train on
        self.backbone = backbone
        self.best_model_path = None
        self.best_model_score = None

    def run(self, train_folder):
        # Create a datamodule from the given dataset
        datamodule = ImageClassificationData.from_folders(
            train_folder=train_folder,
            batch_size=1,
            val_split=0.5,
        )
        # Create an image classfier task with the given backbone
        model = ImageClassifier(datamodule.num_classes, backbone=self.backbone)
        # Start a Lightning trainer, with 1 training batch and 4 validation batches
        trainer = flash.Trainer(
            max_epochs=self.max_epochs,
            limit_train_batches=1,
            limit_val_batches=4,
            callbacks=[ModelCheckpoint(monitor="val_cross_entropy")],
        )
        # Train the model
        trainer.fit(model, datamodule=datamodule)
        # Save the model path
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        # Save the model score
        self.best_model_score = trainer.checkpoint_callback.best_model_score.item()


# Step 2: Create a serving LightningWork component that gets a model input and serves it
class ImageClassifierServeWork(LightningWork):
    def run(self, best_model_path: str):
        # Load the model from the model path
        model = ImageClassifier.load_from_checkpoint(best_model_path)
        model.serve(output="labels")


# Step 3: Create a root LightningFlow component that gets number of epochs and a path to
# a dataset as inputs, initialize 2 training components and serves the best model
class RootFlow(LightningFlow):
    def __init__(self, max_epochs: int, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        # Init an image classifier with resnet18 backbone
        self.train_work_1 = ImageClassifierTrainWork(
            max_epochs,
            "resnet18",
        )
        # Init an image classifier with resnet26 backbone
        self.train_work_2 = ImageClassifierTrainWork(
            max_epochs,
            "resnet26",
        )
        # Init the serving component
        self.server_work = ImageClassifierServeWork()

    def run(self):
        # running both `train_work_1` and `train_work_2` in parallel and asynchronously.
        self.train_work_1.run(self.data_dir)
        self.train_work_2.run(self.data_dir)

        # run serve_work only when both `best_model_score` are available.
        if self.train_work_1.best_model_score and self.train_work_2.best_model_score:
            # serve only the best model between `train_work_1` and `train_work_2`.
            self.server_work.run(
                self.train_work_1.best_model_path
                if self.train_work_1.best_model_score < self.train_work_2.best_model_score
                else self.train_work_2.best_model_path
            )


# Step 4: download a dataset to your local directory under `/data`
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

# Initialize your Lightning app with 5 epochs
app = LightningApp(RootFlow(5, "./data/hymenoptera_data"))
