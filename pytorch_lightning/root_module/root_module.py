import torch

from pytorch_lightning.root_module.memory import ModelSummary
from pytorch_lightning.root_module.grads import GradInformation
from pytorch_lightning.root_module.model_saving import ModelIO, load_hparams_from_tags_csv
from pytorch_lightning.root_module.hooks import ModelHooks
from pytorch_lightning.root_module.decorators import data_loader


class LightningModule(GradInformation, ModelIO, ModelHooks):

    def __init__(self, *args, **kwargs):
        super(LightningModule, self).__init__(*args, **kwargs)

        self.dtype = torch.FloatTensor
        self.exp_save_path = None
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.experiment = None
        self.example_input_array = None

        # track if gpu was requested for checkpointing
        self.on_gpu = False

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
        Return a list of optimizers and a list of schedulers (could be empty)
        :return:
        """
        raise NotImplementedError

    @data_loader
    def tng_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

    @data_loader
    def test_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

    @data_loader
    def val_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

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

        # load the state_dict on the model automatically
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    def summarize(self):
        model_summary = ModelSummary(self)
        print(model_summary)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
