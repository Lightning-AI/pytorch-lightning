import torch
import torchvision.transforms as T


class ToUint8:
    def __call__(self, x):
        return (x * 255.0).type(torch.uint8)


class ToFloat:
    def __call__(self, x):
        return x.float()


TRANSFORMS = {
    "simple": T.Compose([T.ToTensor(), T.Resize((196, 196))]),
    "rotation": T.Compose([T.ToTensor(), T.Resize((196, 196)), T.RandomRotation(45)]),
    "randaugment": T.Compose([T.ToTensor(), T.Resize((196, 196)), ToUint8(), T.RandAugment(), ToFloat()]),
}
