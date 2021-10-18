"""
DCGAN - Adapted from pytorch/examples

Launch it with this command:

python -m torch.distributed.run --nproc_per_node=2 gan_example.py

"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DistributedSampler

from pl_examples.lite_examples.gan.models import Discriminator, Generator, weights_init
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteModule, _LiteOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, help="number of data loading workers", default=0)
parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
parser.add_argument(
    "--imageSize",
    type=int,
    default=64,
    help="the height / width of the input image to network",
)
parser.add_argument("--niter", type=int, default=25, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate, default=0.0002")
parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--netG", default="", help="path to netG (to continue training)")
parser.add_argument("--netD", default="", help="path to netD (to continue training)")
parser.add_argument("--outf", default="./lightning_logs", help="folder to output images and model checkpoints")
parser.add_argument("--local_rank", type=int, default=0)

opt, _ = parser.parse_known_args()
os.makedirs(opt.outf, exist_ok=True)
ngpu = int(opt.ngpu)

nz = 100


class GANTrainer(LightningLite):
    def run(self):
        print("strategy: ", self._strategy)
        print("precision plugin: ", self._precision_plugin)
        seed_everything(123)

        # TODO: how do we handle this in Accelerator?
        # torch.cuda.set_device(opt.local_rank)
        # TODO: how do we handle this?
        # os.environ["LOCAL_RANK"] = str(opt.local_rank)
        # os.environ["NODE_RANK"] = str(opt.local_rank)

        if self.local_rank == 0:
            dset.MNIST(root=".", download=True)

        self.barrier()
        dataset = dset.MNIST(
            root=".",
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers
        )

        dataloader = self.setup_dataloaders(dataloader)
        # assert isinstance(dataloader.sampler, DistributedSampler)

        netG = Generator()
        netG.apply(weights_init)

        netD = Discriminator()
        netD.apply(weights_init)

        # self.to_device(netG)
        # self.to_device(netD)

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=self.device)
        real_label = 1
        fake_label = 0

        # setup optimizer
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        netG, optimizerG = self.setup(netG, optimizerG)
        netD, optimizerD = self.setup(netD, optimizerD)

        assert isinstance(optimizerG, _LiteOptimizer)
        assert isinstance(netG, _LiteModule)
        print("parameters dtype", next(netG.parameters()).dtype)

        for epoch in range(opt.niter):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data[0]
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=self.device)

                output = netD(real_cpu)
                errD_real = criterion(output, label)
                self.backward(errD_real)
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=self.device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                self.backward(errD_fake)
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                self.backward(errG)
                D_G_z2 = output.mean().item()
                optimizerG.step()

                print(
                    "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        opt.niter,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )
                if i % 100 == 0:
                    vutils.save_image(real_cpu, "%s/real_samples.png" % opt.outf, normalize=True)
                    fake = netG(fixed_noise)
                    vutils.save_image(
                        fake.detach(),
                        "%s/fake_samples_epoch_%03d.png" % (opt.outf, epoch),
                        normalize=True,
                    )
            # do checkpointing
            torch.save(netG.state_dict(), "%s/netG_epoch_%d.pth" % (opt.outf, epoch))
            torch.save(netD.state_dict(), "%s/netD_epoch_%d.pth" % (opt.outf, epoch))


if __name__ == "__main__":
    gan = GANTrainer(accelerator="ddp", devices=2)
    gan.run()
