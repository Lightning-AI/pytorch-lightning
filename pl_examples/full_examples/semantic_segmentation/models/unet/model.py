import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet.parts import DoubleConv, Down, Up


class UNet(nn.Module):
    '''
    Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link - https://arxiv.org/abs/1505.04597

    Parameters:
    num_classes (int) - Number of output classes required (default 19 for KITTI dataset)
    bilinear (bool) - Whether to use bilinear interpolation or transposed
    convolutions for upsampling.
    '''
    def __init__(self, num_classes=19, bilinear=False):
        super().__init__()
        self.layer1 = DoubleConv(3, 64)
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        self.layer5 = Down(512, 1024)

        self.layer6 = Up(1024, 512, bilinear=bilinear)
        self.layer7 = Up(512, 256, bilinear=bilinear)
        self.layer8 = Up(256, 128, bilinear=bilinear)
        self.layer9 = Up(128, 64, bilinear=bilinear)

        self.layer10 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.layer6(x5, x4)
        x6 = self.layer7(x6, x3)
        x6 = self.layer8(x6, x2)
        x6 = self.layer9(x6, x1)

        return self.layer10(x6)
