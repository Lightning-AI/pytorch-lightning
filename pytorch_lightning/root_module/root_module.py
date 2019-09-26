import torch

from pytorch_lightning.root_module.memory import ModelSummary
from pytorch_lightning.root_module.grads import GradInformation
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.root_module.model_saving import ModelIO
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
        self.use_dp = False
        self.use_ddp = False
        self.use_amp = False

    def forward(self, *args, **kwargs):
        """
        Expand model in into whatever you need.
        Also need to return the target
        :param x:
        :return:
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        """
        return loss, dict with metrics for tqdm
        :param called with batch, batch_nb
        additional: optimizer_i if multiple optimizers used
        :return:
        """
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):
        """
        return whatever outputs will need to be aggregated in validation_end
        OPTIONAL
        :param called with batch, batch_nb
        additional: dataset_i if multiple val datasets used
        :return:
        """
        pass

    def test_step(self, *args, **kwargs):
        """
        return whatever outputs will need to be aggregated in test_end
        OPTIONAL
        :param called with batch, batch_nb
        additional: dataset_i if multiple val datasets used
        :return:
        """
        pass

    def validation_end(self, outputs):
        """
        Outputs has the appended output after each validation step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        pass

    def test_end(self, outputs):
        """
        Outputs has the appended output after each test step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        pass

    def configure_optimizers(self):
        """
        Return a list of optimizers and a list of schedulers (could be empty)
        :return:
        """
        raise NotImplementedError

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i):
        """
        Do something instead of the standard optimizer behavior
        :param epoch_nb:
        :param batch_nb:
        :param optimizer:
        :param optimizer_i:
        :return:
        """
        optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    @data_loader
    def train_dataloader(self):
        """
        Implement a PyTorch DataLoader
        :return:
        """
        raise NotImplementedError

    @data_loader
    def test_dataloader(self):
        """
        Implement a PyTorch DataLoader
        :return:
        """
        return None

    @data_loader
    def val_dataloader(self):
        """
        Implement a PyTorch DataLoader
        :return:
        """
        return None

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv, on_gpu):
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

        # load on CPU only to avoid OOM issues
        # then its up to user to put back on GPUs
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
