import torch.nn as nn
import numpy as np
from pytorch_lightning.root_module.root_module import LightningModule
from test_tube import HyperOptArgumentParser
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F


class ExampleModel1(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams):
        # init superclass
        super(ExampleModel1, self).__init__(hparams)

        self.batch_size = hparams.batch_size

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.c_d1 = nn.Linear(in_features=self.hparams.in_features, out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim, out_features=self.hparams.out_features)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        x = self.c_d1(x)
        x = F.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)

        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def training_step(self, data_batch):
        """
        Called inside the training loop
        :param data_batch:
        :return:
        """
        # forward pass
        x, y = data_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        tqdm_dic = {'jefe': 1}
        return loss_val, tqdm_dic

    def validation_step(self, data_batch):
        """
        Called inside the validation loop
        :param data_batch:
        :return:
        """
        x, y = data_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        output = {'y_hat': y_hat, 'val_loss': loss_val.item(), 'val_acc': val_acc}
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        val_loss_mean = 0
        accs = []
        for output in outputs:
            val_loss_mean += output['val_loss']
            accs.append(output['val_acc'])

        val_loss_mean /= len(outputs)
        tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': np.mean(accs)}
        return tqdm_dic

    def update_tng_log_metrics(self, logs):
        return logs

    # ---------------------
    # MODEL SAVING
    # ---------------------
    def get_save_dict(self):
        checkpoint = {
            'state_dict': self.state_dict(),
        }

        return checkpoint

    def load_model_specific(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'])
        pass

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = self.choose_optimizer(self.hparams.optimizer_name, self.parameters(), {'lr': self.hparams.learning_rate}, 'optimizer')
        self.optimizers = [optimizer]
        return self.optimizers

    def __dataloader(self, train):
        # init data generators
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        dataset = MNIST(root=self.hparams.data_root, train=train, transform=transform, download=True)

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )

        return loader

    @property
    def tng_dataloader(self):
        if self._tng_dataloader is None:
            try:
                self._tng_dataloader = self.__dataloader(train=True)
            except Exception as e:
                print(e)
                raise e
        return self._tng_dataloader

    @property
    def val_dataloader(self):
        if self._val_dataloader is None:
            try:
                self._val_dataloader = self.__dataloader(train=False)
            except Exception as e:
                print(e)
                raise e
        return self._val_dataloader

    @property
    def test_dataloader(self):
        if self._test_dataloader is None:
            try:
                self._test_dataloader = self.__dataloader(train=False)
            except Exception as e:
                print(e)
                raise e
        return self._test_dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip=5.0)

        # network params
        parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=False)
        parser.add_argument('--in_features', default=28*28)
        parser.add_argument('--hidden_dim', default=500)
        parser.add_argument('--out_features', default=10)

        # data
        parser.add_argument('--data_root', default='/Users/williamfalcon/Developer/personal/research_lib/research_proj/datasets/mnist', type=str)

        # training params (opt)
        parser.opt_list('--learning_rate', default=0.001, type=float, options=[0.0001, 0.0005, 0.001, 0.005],
                        tunable=False)
        parser.opt_list('--batch_size', default=256, type=int, options=[32, 64, 128, 256], tunable=False)
        parser.opt_list('--optimizer_name', default='adam', type=str, options=['adam'], tunable=False)
        return parser
