import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import fcn_resnet50

from PIL import Image
import pytorch_lightning as pl


class KITTI(Dataset):
    def __init__(self, root_path, split='test', img_size=(1242, 376), transform=None):
        self.img_size = img_size
        self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(19)))
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
    def __init__(self, hparams):
        super(SegModel, self).__init__()
        self.root_path = hparams.root
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.lr
        self.net = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=19)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753], std=[0.32064945, 0.32098866, 0.32325324])
        ])
        self.trainset = KITTI(self.root_path, split='train', transform=self.transform)
        self.testset = KITTI(self.root_path, split='test', transform=self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_val = F.cross_entropy(out['out'], mask, ignore_index=250)
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
    # trainer.fit(model)

    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, help="path where dataset is stored")
    parser.add_argument("--gpus", type=int, help="number of available GPUs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")

    hparams = parser.parse_args()

    main(hparams)
