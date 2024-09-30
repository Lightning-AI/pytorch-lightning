# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lightning.pytorch import LightningModule, Trainer, cli_lightning_logo
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_VOID_LABELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)


def _create_synth_kitti_dataset(path_dir: str, image_dims: tuple = (1024, 512)):
    """Create synthetic dataset with random images, just to simulate that the dataset have been already downloaded."""
    path_dir_images = os.path.join(path_dir, KITTI.IMAGE_PATH)
    path_dir_masks = os.path.join(path_dir, KITTI.MASK_PATH)
    for p_dir in (path_dir_images, path_dir_masks):
        os.makedirs(p_dir, exist_ok=True)
    for i in range(3):
        path_img = os.path.join(path_dir_images, f"dummy_kitti_{i}.png")
        Image.new("RGB", image_dims).save(path_img)
        path_mask = os.path.join(path_dir_masks, f"dummy_kitti_{i}.png")
        Image.new("L", image_dims).save(path_mask)


class KITTI(Dataset):
    """Class for KITTI Semantic Segmentation Benchmark dataset.

    Dataset link - http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

    There are 34 classes in the given labels. However, not all of them are useful for training
    (like railings on highways, road dividers, etc.).
    So, these useless classes (the pixel values of these classes) are stored in the `void_labels`.
    The useful classes are stored in the `valid_labels`.

    The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
    (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
    `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
    the loss function when comparing with the output.

    The `get_filenames` function retrieves the filenames of all images in the given `path` and
    saves the absolute path in a list.

    In the `get_item` function, images and masks are resized to the given `img_size`, masks are
    encoded using `encode_segmap`, and given `transform` (if any) are applied to the image only
    (mask does not usually require transforms, but they can be implemented in a similar way).

    """

    IMAGE_PATH = os.path.join("training", "image_2")
    MASK_PATH = os.path.join("training", "semantic")

    def __init__(
        self,
        data_path: str,
        split: str,
        img_size: tuple = (1242, 376),
        void_labels: list = DEFAULT_VOID_LABELS,
        valid_labels: list = DEFAULT_VALID_LABELS,
        transform=None,
    ):
        self.img_size = img_size
        self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.split = split
        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, self.IMAGE_PATH)
        self.mask_path = os.path.join(self.data_path, self.MASK_PATH)
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        # Split between train and valid set (80/20)
        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        if self.split == "train":
            idxs = [idx for idx in range(n_items) if idx not in idxs]
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = torch.tensor(img)

        mask = Image.open(self.mask_list[idx]).convert("L")
        mask = mask.resize(self.img_size)
        mask = torch.tensor(mask)
        mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask

    def encode_segmap(self, mask):
        """Sets void classes to zero so they won't be considered for training."""
        for voidc in self.void_labels:
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        mask[mask > 18] = self.ignore_index
        return mask

    def get_filenames(self, path):
        """Returns a list of absolute paths to images inside given `path`"""
        files_list = []
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class UNet(nn.Module):
    """Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation.

    Link - https://arxiv.org/abs/1505.04597

    >>> UNet(num_classes=2, num_layers=3)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    UNet(
      (layers): ModuleList(
        (0): DoubleConv(...)
        (1): Down(...)
        (2): Down(...)
        (3): Up(...)
        (4): Up(...)
        (5): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
      )
    )

    """

    def __init__(self, num_classes: int = 19, num_layers: int = 5, features_start: int = 64, bilinear: bool = False):
        """
        Args:
            num_classes: Number of output classes required (default 19 for KITTI dataset)
            num_layers: Number of layers in each side of U-net
            features_start: Number of features in first layer
            bilinear: Whether to use bilinear interpolation or transposed convolutions for upsampling.
        """
        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(3, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """Double Convolution and BN and ReLU (3x3 conv -> BN -> ReLU) ** 2.

    >>> DoubleConv(4, 4)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DoubleConv(
      (net): Sequential(...)
    )

    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Combination of MaxPool2d and DoubleConv in series.

    >>> Down(4, 8)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Down(
      (net): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): DoubleConv(
          (net): Sequential(...)
        )
      )
    )

    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature map
    from contracting path, followed by double 3x3 convolution.

    >>> Up(8, 4)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Up(
      (upsample): ConvTranspose2d(8, 4, kernel_size=(2, 2), stride=(2, 2))
      (conv): DoubleConv(
        (net): Sequential(...)
      )
    )

    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegModel(LightningModule):
    """Semantic Segmentation Module.

    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    It uses the FCN ResNet50 model as an example.

    Adam optimizer is used along with Cosine Annealing learning rate scheduler.

    >>> dataset_path = os.path.join(".", "Kitti")
    >>> _create_synth_kitti_dataset(dataset_path, image_dims=(1024, 512))
    >>> SegModel(dataset_path)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    SegModel(
      (net): UNet(
        (layers): ModuleList(
          (0): DoubleConv(...)
          (1): Down(...)
          (2): Down(...)
          (3): Up(...)
          (4): Up(...)
          (5): Conv2d(64, 19, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )

    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 4,
        lr: float = 1e-3,
        num_layers: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.net = UNet(
            num_classes=19, num_layers=self.num_layers, features_start=self.features_start, bilinear=self.bilinear
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753], std=[0.32064945, 0.32098866, 0.32325324]),
        ])
        self.trainset = KITTI(self.data_path, split="train", transform=self.transform)
        self.validset = KITTI(self.data_path, split="valid", transform=self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask, ignore_index=250)
        log_dict = {"train_loss": loss}
        return {"loss": loss, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        val_loss = F.cross_entropy(out, mask, ignore_index=250)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)


def main(hparams: Namespace):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModel(**vars(hparams))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer()

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    cli_lightning_logo()

    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path where dataset is stored")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
    )
    hparams = parser.parse_args()

    main(hparams)
