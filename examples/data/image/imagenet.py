import os
import traceback
from argparse import ArgumentParser
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import lightning as L
from lightning.pytorch.utilities.model_helpers import get_torchvision_model

parser = ArgumentParser()
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--batchsize", default=56, type=int)
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
args = parser.parse_args()

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------


class ImageNetLightningModel(L.LightningModule):
    """
    >>> ImageNetLightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ImageNetLightningModel(
      (model): ResNet(...)
    )
    """

    from torchvision.models.resnet import ResNet18_Weights

    def __init__(
        self,
        data_path: str,
        index_file_path: str = None,
        arch: str = "resnet18",
        weights=ResNet18_Weights.IMAGENET1K_V1,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        workers: int = 4,
    ):
        super().__init__()
        self.arch = arch
        self.weights = weights
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers
        self.data_path = data_path
        self.index_file_path = index_file_path
        self.model = get_torchvision_model(self.arch, weights=self.weights)
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(f"{prefix}_loss", loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        import torchvision as tv

        transforms = tv.transforms.Compose([tv.transforms.RandomResizedCrop(224), tv.transforms.ToTensor()])

        train_dataset = S3LightningImagenetDataset(
            data_source=self.data_path, split="train", transforms=transforms, path_to_index_file=self.index_file_path
        )

        return torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers
        )

    def val_dataloader(self):
        import torchvision as tv

        transforms = tv.transforms.Compose([tv.transforms.RandomResizedCrop(224), tv.transforms.ToTensor()])

        val_dataset = S3LightningImagenetDataset(
            data_source=self.data_path, split="val", transforms=transforms, path_to_index_file=self.index_file_path
        )

        return torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers
        )

    def test_dataloader(self):
        return self.val_dataloader()


# -------------------
# Step 2: Define data
# -------------------


class S3LightningImagenetDataset(L.LightningDataset):
    def __init__(
        self,
        data_source: str,
        split: Literal["train", "val"],
        transforms: Optional[Callable] = None,
        path_to_index_file: Optional[str] = None,
    ):
        from torchvision.models._meta import _IMAGENET_CATEGORIES

        super().__init__(data_source=data_source, backend="s3", path_to_index_file=path_to_index_file)

        # only get files for the split
        self.files = tuple([x for x in self.files if split in x])

        # get unique classes
        self.classes = _IMAGENET_CATEGORIES

        self.transforms = transforms

    def load_sample(self, file_path, stream):
        from PIL import Image

        try:
            img = Image.open(stream)

            if self.transforms is not None:
                img = self.transforms(img)

            # Converting grey scale images to RGB
            if img.shape[0] == 1:
                img = img.repeat((3, 1, 1))

            curr_cls = os.path.basename(os.path.dirname(file_path)).replace("_", " ")
            cls_idx = self.classes.index(curr_cls)
            return img, cls_idx
        except Exception:
            print(file_path, traceback.print_exc())
            pass


if __name__ == "__main__":
    # os.environ["AWS_ACCESS_KEY"] = <your aws access key>
    # os.environ["AWS_SECRET_ACCESS_KEY"] = <your aws secret key>

    data_path = "s3://imagenet-tiny"
    index_file_path = "imagenet/imagenet-index.txt"

    # -------------------
    # Step 3: Train
    # -------------------

    print("Instantiate Model")
    model = ImageNetLightningModel(
        weights=None,
        data_path=data_path,
        index_file_path=index_file_path,
        batch_size=args.batchsize,
        workers=args.workers,
    )
    trainer = L.Trainer()

    print("Train Model")
    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)
