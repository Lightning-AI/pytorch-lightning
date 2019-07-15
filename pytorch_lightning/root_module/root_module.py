import os
import torch
import math

from pytorch_lightning.root_module.memory import ModelSummary
from pytorch_lightning.root_module.grads import GradInformation
from pytorch_lightning.root_module.model_saving import ModelIO, load_hparams_from_tags_csv
from pytorch_lightning.root_module.optimization import OptimizerConfig
from pytorch_lightning.root_module.hooks import ModelHooks


class LightningModule(GradInformation, ModelIO, OptimizerConfig, ModelHooks):

    def __init__(self, hparams):
        super(LightningModule, self).__init__()
        self.hparams = hparams

        self.dtype = torch.FloatTensor
        self.exp_save_path = None
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.experiment = None

        # track if gpu was requested for checkpointing
        self.on_gpu = False

        # computed vars for the dataloaders
        self._tng_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def forward(self, *args, **kwargs):
        """
        Expand model in into whatever you need.
        Also need to return the target
        :param x:
        :return:
        """
        raise NotImplementedError

    def validation_step(self, data_batch, batch_nb):
        """
        return whatever outputs will need to be aggregated in validation_end
        :param data_batch:
        :return:
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        """
        Outputs has the appended output after each validation step
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        raise NotImplementedError

    def training_step(self, data_batch, batch_nb):
        """
        return loss, dict with metrics for tqdm
        :param data_batch:
        :return:
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """
        Return array of optimizers
        :return:
        """
        raise NotImplementedError

    def update_tng_log_metrics(self, logs):
        """
        Chance to update metrics to be logged for training step.
        For example, add music, images, etc... to log
        :param logs:
        :return:
        """
        return logs

    def loss(self, *args, **kwargs):
        """
        Expand model_out into your components
        :param model_out:
        :return:
        """
        raise NotImplementedError

    def summarize(self):
        model_summary = ModelSummary(self)
        print(model_summary)


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def tng_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

    @property
    def test_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

    @property
    def val_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def get_process_position(gpus):
        try:
            current_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
            gpu_ids = gpus.split(',')
            process_position = gpu_ids.index(current_gpu)
            return process_position, current_gpu
        except Exception as e:
            return 0, 0

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv, on_gpu, map_location=None):
        """
        Primary way of loading model from csv weights path
        :param weights_path:
        :param tags_csv:
        :param on_gpu:
        :param map_location: dic for mapping storage {'cuda:1':'cuda:0'}
        :return:
        """
        hparams = load_hparams_from_tags_csv(tags_csv)
        hparams.__setattr__('on_gpu', on_gpu)

        if on_gpu:
            if map_location is not None:
                checkpoint = torch.load(weights_path, map_location=map_location)
            else:
                checkpoint = torch.load(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        model = cls(hparams)

        # allow model to load
        model.load_model_specific(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model
