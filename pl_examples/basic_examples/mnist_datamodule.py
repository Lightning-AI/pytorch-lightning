import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import MNIST
except ModuleNotFoundError:
    from tests.base.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """
    Standard MNIST, train, val, test splits and transforms

    Example::

        from pl_bolts.datamodules import MNISTDataModule

        dm = MNISTDataModule('.')
        model = LitModel()

        Trainer().fit(model, dm)
    """

    name = "mnist"

    def __init__(
        self,
        data_dir: str = "./",
        val_split: int = 5000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
        """
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves MNIST files to data_dir
        """
        dataset = MNIST(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        _ = MNIST(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())

        train_length = len(dataset)
        self.dataset_train, self.dataset_val = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        """
        MNIST train set removes a subset to use for validation
        """
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """
        MNIST val set uses a subset of the training set for validation
        """
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        """
        MNIST test set uses the test split
        """
        dataset = MNIST(self.data_dir, train=False, download=False, transform=transform_lib.ToTensor())
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True,
            pin_memory=True
        )
        return loader
