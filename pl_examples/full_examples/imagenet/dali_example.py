"""
This example is to demonstrate the usage of a nvidia-dali pipeline inside pl.
For further details,
please refer to [nvidia-dali document](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)
"""

import os
import pytorch_lightning as pl
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from imagenet_example import ImageNetLightningModel, get_args, main
from dali_utils import HybridTrainPipe, HybridValPipe


class DaliImageNetLightningModel(ImageNetLightningModel):
    def split_batch(self, batch):
        image = batch[0]["data"]
        target = batch[0]["label"].squeeze().long()
        return image, target

    @pl.data_loader
    def train_dataloader(self):
        train_dir = os.path.join(self.hparams.data_path, "train")
        pipe = HybridTrainPipe(batch_size=self.hparams.batch_size, num_threads=2,
                               local_rank=self.trainer.proc_rank, world_size=self.trainer.world_size,
                               data_dir=train_dir)
        pipe.build()
        pipe_size = int(pipe.epoch_size("Reader") / self.trainer.world_size)
        return DALIClassificationIterator(pipe, size=pipe_size, auto_reset=True)

    @pl.data_loader
    def val_dataloader(self):
        val_dir = os.path.join(self.hparams.data_path, "val")
        pipe = HybridValPipe(batch_size=self.hparams.batch_size, num_threads=2,
                             local_rank=self.trainer.proc_rank, world_size=self.trainer.world_size,
                             data_dir=val_dir)
        pipe.build()
        pipe_size = int(pipe.epoch_size("Reader") / self.trainer.world_size)
        return DALIClassificationIterator(pipe, size=pipe_size, auto_reset=True)


if __name__ == '__main__':
    main(get_args(), pl_model=DaliImageNetLightningModel)
