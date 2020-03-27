import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from models.unet.model import UNet
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl


class KITTI(Dataset):
    '''
    Dataset Class for KITTI Semantic Segmentation Benchmark dataset
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
    '''

    def __init__(
        self,
        root_path,
        split='test',
        img_size=(1242, 376),
        void_labels=[0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1],
        valid_labels=[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        transform=None
    ):
        self.img_size = img_size
        self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.split = split
        self.root = root_path
        if self.split == 'train':
            self.img_path = os.path.join(self.root, 'training/image_2')
            self.mask_path = os.path.join(self.root, 'training/semantic')
        else:
            self.img_path = os.path.join(self.root, 'testing/image_2')
            self.mask_path = None

        self.transform = transform

        self.img_list = self.get_filenames(self.img_path)
        if self.split == 'train':
            self.mask_list = self.get_filenames(self.mask_path)
        else:
            self.mask_list = None

    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        if self.split == 'train':
            mask = Image.open(self.mask_list[idx]).convert('L')
            mask = mask.resize(self.img_size)
            mask = np.array(mask)
            mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)

        if self.split == 'train':
            return img, mask
        else:
            return img

    def encode_segmap(self, mask):
        '''
        Sets void classes to zero so they won't be considered for training
        '''
        for voidc in self.void_labels:
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        return mask

    def get_filenames(self, path):
        '''
        Returns a list of absolute paths to images inside given `path`
        '''
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class SegModel(pl.LightningModule):
    '''
    Semantic Segmentation Module

    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    It uses the FCN ResNet50 model as an example.

    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    '''

    def __init__(self, hparams):
        super().__init__()
        self.root_path = hparams.root
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.lr
        self.net = UNet(num_classes=19)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])
        self.trainset = KITTI(self.root_path, split='train', transform=self.transform)
        self.testset = KITTI(self.root_path, split='test', transform=self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        return {'loss': loss_val}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=hparams.gpus
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, help="path where dataset is stored")
    parser.add_argument("--gpus", type=int, help="number of available GPUs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")

    hparams = parser.parse_args()

    main(hparams)
