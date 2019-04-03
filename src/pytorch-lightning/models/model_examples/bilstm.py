import torch.nn as nn
import numpy as np

from test_tube import HyperOptArgumentParser
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F


class BiLSTMPack(nn.Module):
    """
    Sample model to show how to define a template
    """
    def __init__(self, hparams):
        # init superclass
        super(BiLSTMPack, self).__init__(hparams)

        self.hidden = None

        # trigger tag building
        self.ner_tagset = {'O': 0, 'I-Bio': 1}
        self.nb_tags = len(self.ner_tagset)

        # build model
        print('building model...')
        if hparams.model_load_weights_path is None:
            self.__build_model()
            print('model built')
        else:
            self = BiLSTMPack.load(hparams.model_load_weights_path, hparams.on_gpu, hparams)
            print('model loaded from: {}'.format(hparams.model_load_weights_path))

    def __build_model(self):
        """
        Layout model
        :return:
        """
        # design the number of final units
        self.output_dim = self.hparams.nb_lstm_units

        # when it's bidirectional our weights double
        if self.hparams.bidirectional:
            self.output_dim *= 2

        # total number of words
        total_words = len(self.tng_dataloader.dataset.words_token_to_idx)

        # word embeddings
        self.word_embedding = nn.Embedding(
            num_embeddings=total_words + 1,
            embedding_dim=self.hparams.embedding_dim,
            padding_idx=0
        )

        # design the LSTM
        self.lstm = nn.LSTM(
            self.hparams.embedding_dim,
            self.hparams.nb_lstm_units,
            num_layers=self.hparams.nb_lstm_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.drop_prob,
            batch_first=True,
        )

        # map to tag space
        self.fc_out = nn.Linear(self.output_dim, self.out_dim)
        self.hidden_to_tag = nn.Linear(self.output_dim, self.nb_tags)


    def init_hidden(self, batch_size):

        # the weights are of the form (nb_layers * 2 if bidirectional, batch_size, nb_lstm_units)
        mult = 2 if self.hparams.bidirectional else 1
        hidden_a = torch.randn(self.hparams.nb_layers * mult, batch_size, self.nb_rnn_units)
        hidden_b = torch.randn(self.hparams.nb_layers * mult, batch_size, self.nb_rnn_units)

        if self.hparams.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, model_in):
        # layout data (expand it, etc...)
        # x = sequences
        x, seq_lengths = model_in
        batch_size, seq_len = x.size()

        # reset RNN hidden state
        self.hidden = self.init_hidden(batch_size)

        # embed
        x = self.word_embedding(x)

        # run through rnn using packed sequences
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
        x, self.hidden = self.lstm(x, self.hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # if asked for only last state, use the h_n which is the same as out(t=n)
        if not self.return_sequence:
            # pull out hidden states
            # h_n = (nb_directions * nb_layers, batch_size, emb_size)
            nb_directions = 2 if self.bidirectional else 1
            (h_n, _) = self.hidden

            # reshape to make indexing easier
            # forward = 0, backward = 1 (of nb_directions)
            h_n = h_n.view(self.nb_layers, nb_directions, batch_size, self.nb_rnn_units)

            # pull out last forward
            forward_h_n = h_n[-1, 0, :, :]
            x = forward_h_n

            # if bidirectional, also pull out the last hidden of backward network
            if self.bidirectional:
                backward_h_n = h_n[-1, 1, :, :]
                x = torch.cat([forward_h_n, backward_h_n], dim=1)

        # project to tag space
        x = x.contiguous()
        x = x.view(-1, self.output_dim)
        x = self.hidden_to_tag(x)

        return x

    def loss(self, model_out):
        # cross entropy loss
        logits, y = model_out
        y, y_lens = y

        # flatten y and logits
        y = y.view(-1)
        logits = logits.view(-1, self.nb_tags)

        # calculate a mask to remove padding tokens
        mask = (y >= 0).float()

        # count how many tokens we have
        num_tokens = int(torch.sum(mask).data[0])

        # pick the correct values and mask out
        logits = logits[range(logits.shape[0]), y] * mask

        # compute the ce loss
        ce_loss = -torch.sum(logits)/num_tokens

        return ce_loss

    def pull_out_last_embedding(self, x, seq_lengths, batch_size, on_gpu):
        # grab only the last activations from the non-padded ouput
        x_last = torch.zeros([batch_size, 1, x.size(-1)])
        for i, seq_len in enumerate(seq_lengths):
            x_last[i, :, :] = x[i, seq_len-1, :]

        # put on gpu when requested
        if on_gpu:
            x_last = x_last.cuda()

        # turn into torch var
        x_last = Variable(x_last)

        return x_last