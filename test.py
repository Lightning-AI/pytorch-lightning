import argparse
from typing import Optional
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics import PSNR
from tests.helpers.datasets import MNIST

def add_gaussian_noise(cleanTensor, sigma=.1):
	# adds gausian noise of standard deviation sigma
	noiseTensor = torch.normal(mean=torch.zeros_like(cleanTensor), std=sigma)
	noisyTensor = cleanTensor + noiseTensor
	return noiseTensor, noisyTensor

class LitConvAE(pl.LightningModule):
	def __init__(self, hparams):
		super().__init__()
		# network architecture
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
			nn.ReLU(True),
			nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1))
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
			nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(True))
		# Model-specific parameters
		self.learning_rate = hparams.learning_rate
		self.noise_sigma = hparams.noise_sigma
		# save all hyperparameters to a .yaml file
		self.save_hyperparameters()
		# metrics from torchmetrics
		self.train_psnr = PSNR(data_range=1, dim=(-2, -1))
		self.val_psnr = PSNR(data_range=1, dim=(-2, -1))

	def forward(self, x):
		# typically defines inference behavior
		x = self.encoder(x)
		x = self.decoder(x)
		return x

	def training_step(self, batch, batch_idx):
		# training behavior can be different from that of inference
		clean_batch, _ = batch  # do not care about image class for denoising
		noise_batch, noisy_batch = add_gaussian_noise(clean_batch, self.noise_sigma)
		denoised_batch = self.decoder(self.encoder(noisy_batch))
		loss = nn.functional.mse_loss(denoised_batch, clean_batch, reduction='sum')  # squared l2 norm
		self.log('train_loss', loss)  # log at each step
		self.train_psnr(denoised_batch, clean_batch)
		self.log('train_psnr', self.train_psnr, on_step=False, on_epoch=True)  # log at each end of epoch
		return loss

	def validation_step(self, batch, batch_idx):
		# training behavior can be different from that of inference
		clean_batch, _ = batch  # do not care about image class for denoising
		noise_batch, noisy_batch = add_gaussian_noise(clean_batch, self.noise_sigma)
		denoised_batch = self.decoder(self.encoder(noisy_batch))
		self.val_psnr(denoised_batch, clean_batch)
		self.log('validation_psnr', self.val_psnr, on_step=False, on_epoch=True)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer

	@staticmethod
	def add_model_specific_args(parent_parser):
		# Model-specific arguments
		parser = parent_parser.add_argument_group("LitConvAE")
		parser.add_argument('--noise_sigma', type=float, default=.2, help='noise standard deviation (between 0. and 1.)')
		parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
		return parent_parser


class MNISTDataModule(pl.LightningDataModule):
	def __init__(self, batch_size=32, dataset_dir='./', data_transform=transforms.ToTensor(), num_workers=4):
		super().__init__()
		self.batch_size = batch_size
		self.dataset_dir = dataset_dir
		self.data_transform = data_transform
		self.num_workers = num_workers

	def prepare_data(self):
		# Use this method to do things that might write to disk
		# or that need to be done only from a single process in distributed settings.
		MNIST(root=self.dataset_dir, train=True, download=True)
		MNIST(root=self.dataset_dir, train=False, download=True)

	def setup(self, stage: Optional[str] = None):
		# data operations you might want to perform on every GPU
		if stage == 'fit' or stage is None:
			dataset_full = MNIST(self.dataset_dir, train=True, transform=self.data_transform)
			train_split = 11 * len(dataset_full) // 12
			print(f"\ntrain / val split: {[train_split, len(dataset_full) - train_split]} \n")
			self.dataset_train, self.dataset_val = random_split(dataset_full, [train_split, len(dataset_full) - train_split])

		# Assign test dataset for use in dataloader(s)
		if stage == 'test' or stage is None:
			self.dataset_test = datasets.MNIST(self.data_dir, train=False, transform=self.data_transform)

	def train_dataloader(self):
		return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

	def val_dataloader(self):
		return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise MNIST with a convolutional Autoencoder")
	# DataModule-specific arguments
	parser.add_argument('--batch_size', type=int, default=32, help='number of examples per batch')
	parser.add_argument('--num_workers', type=int, default=4, help='number of separate processes for the DataLoader (default: 4)')
	# Trainer arguments
	parser.add_argument('--gpus', type=int, default=2, help='how many gpus to select')
	parser.add_argument('--accelerator', type=str, default='ddp', help="which multi-GPU backend you want to use (default: 'ddp')")
	parser.add_argument('--max_epochs', type=int, default=10, help='number of epochs you want the model to train for')
	# Program-specific arguments
	parser.add_argument('--data_dir', type=str, default=r'path/to_mnist_dir', help='path to the parent directory of MNIST torchvision dataset')

	# add model specific args
	parser = LitConvAE.add_model_specific_args(parser)

	hyperparams = parser.parse_args()

	# initialize the neural network
	model = LitConvAE(hyperparams)

	dataModule = MNISTDataModule(batch_size=hyperparams.batch_size, dataset_dir=hyperparams.data_dir, data_transform=transforms.ToTensor())

	trainer = pl.Trainer.from_argparse_args(hyperparams)

	# the training and validation loops happen here
	trainer.fit(model, dataModule)