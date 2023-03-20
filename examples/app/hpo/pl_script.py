import argparse
import os

import pandas as pd
import torch
from flash import Trainer
from flash.image import ImageClassificationData, ImageClassifier

# Parse arguments provided by the Work.
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--submission_path", type=str, required=True)
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--best_model_path", type=str, required=True)
# Optional
parser.add_argument("--backbone", type=str, default="resnet18")
parser.add_argument("--learning_rate", type=float, default=0.01)
args = parser.parse_args()


datamodule = ImageClassificationData.from_folders(
    train_folder=args.train_data_path,
    batch_size=8,
)

model = ImageClassifier(datamodule.num_classes, backbone=args.backbone)
trainer = Trainer(fast_dev_run=True)
trainer.fit(model, datamodule=datamodule)
trainer.save_checkpoint(args.best_model_path)

datamodule = ImageClassificationData.from_folders(
    predict_folder=args.test_data_path,
    batch_size=8,
)

predictions = Trainer().predict(model, datamodule=datamodule)
submission_data = [
    {"filename": os.path.basename(p["metadata"]["filepath"]), "label": torch.argmax(p["preds"]).item()}
    for batch in predictions
    for p in batch
]
df = pd.DataFrame(submission_data)
df.to_csv(args.submission_path, index=False)
