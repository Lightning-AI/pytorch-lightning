# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchtext
from torchtext.datasets import text_classification

NGRAMS = 2

'''
Pytorch equivalent 
https://github.com/pytorch/tutorials/blob/master/beginner_source/text_sentiment_ngrams_tutorial.py
'''
class LitTextSentiment(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, num_class, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


    def training_step(self, batch, batch_idx):
        x, offset, y = batch
        #x,offset,y = self.generate_batch(batch)
        #y_hat = self(x)
        y_hat = self(x,offset)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x,offset,y = batch
        #x,offset,y = self.generate_batch(batch)
        #y_hat = self(x)
        y_hat = self(x,offset)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x,offset,y = batch
        #x,offset,y = self.generate_batch(batch)
        #y_hat = self(x)
        y_hat = self(x,offset)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def generate_batch(self, batch):
        label = torch.tensor([entry[0] for entry in batch])
        text = [entry[1] for entry in batch]
        offsets = [0] + [len(entry) for entry in text]
        # torch.Tensor.cumsum returns the cumulative sum
        # of elements in the dimension dim.

        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text, offsets, label

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def run():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    #parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTextSentiment.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------

    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./.data', ngrams=NGRAMS, vocab=None)
    #print(len(train_dataset)) 
    X_train, X_val = random_split(train_dataset, [100000, 20000])

    train_loader = DataLoader(X_train, batch_size=args.batch_size)
    val_loader = DataLoader(X_val, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = 32
    NUN_CLASS = len(train_dataset.get_labels())

    # ------------
    # model
    # ------------
    model = LitTextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS, args.learning_rate)

    # ------------
    # training
    # ------------
    #trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer(fast_dev_run=True, weights_summary='full')
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    run()