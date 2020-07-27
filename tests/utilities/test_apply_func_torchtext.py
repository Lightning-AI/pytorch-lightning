import torch
import torchtext
from torchtext.data.example import Example

import pytorch_lightning as pl
from pytorch_lightning.utilities.apply_func import move_data_to_device


def _get_torchtext_data_iterator(include_lengths=False):
    text_field = torchtext.data.Field(sequential=True, pad_first=False,  # nosec
                                      init_token="<s>", eos_token="</s>",  # nosec
                                      include_lengths=include_lengths)  # nosec

    example1 = Example.fromdict({"text": "a b c a c"}, {"text": ("text", text_field)})
    example2 = Example.fromdict({"text": "b c a a"}, {"text": ("text", text_field)})
    example3 = Example.fromdict({"text": "c b a"}, {"text": ("text", text_field)})

    dataset = torchtext.data.Dataset([example1, example2, example3],
                                     {"text": text_field}
                                     )
    text_field.build_vocab(dataset)

    iterator = torchtext.data.Iterator(dataset, batch_size=3,
                                       sort_key=None, device=None,
                                       batch_size_fn=None,
                                       train=True, repeat=False, shuffle=None,
                                       sort=None, sort_within_batch=None)
    return iterator, text_field


def test_torchtext_field_include_length_true():
    """Test if batches created by torchtext with include_lengths=True raise an exception."""

    class DebugModel(pl.LightningModule):

        def __init__(self):
            super(DebugModel, self).__init__()

            # setup data loader generating batches with fields consisting of tuples of tensors
            self.debug_data_loader, self.text_field = _get_torchtext_data_iterator(include_lengths=True)

            self.learning_rate = 0.001

            pad_idx = self.text_field.vocab.stoi['<pad>']
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

            self.INPUT_DIM = len(self.text_field.vocab)
            self.ENC_EMB_DIM = 4  # keep it small for debugging
            self.embedding = torch.nn.Embedding(self.INPUT_DIM, self.ENC_EMB_DIM)

            self.hid_dim = 4
            self.rnn = torch.nn.GRU(self.ENC_EMB_DIM, self.hid_dim, 1, bidirectional=False)
            self.out = torch.nn.Linear(self.hid_dim, self.embedding.num_embeddings)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def forward(self, input_seq, length):
            embedded = self.embedding(input_seq)
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded,
                                                                      length,
                                                                      batch_first=False,
                                                                      enforce_sorted=False)
            packed_outputs, hidden = self.rnn(packed_embedded)
            outputs, length = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs)

            output = outputs.squeeze(0)
            prediction = self.out(output)

            return prediction

        @staticmethod
        def _parse_batch(batch):
            source = batch.text[0]
            source_length = batch.text[1]

            return source, source_length

        def training_step(self, batch, batch_nb):
            """ Needed for testing data transfer. """
            x = self._parse_batch(batch)
            target, target_length = x

            output = self.forward(target, target_length)
            loss = self.criterion(output[:-1].view(-1, output.shape[2]), target[1:].view(-1))

            prefix = 'train'
            tensorboard_logs = {f'{prefix}_loss': loss.item()}

            result = {'loss': loss, 'log': tensorboard_logs}
            return result

        def train_dataloader(self):
            return self.debug_data_loader

    model = DebugModel()

    cuda_device_cnt = torch.cuda.device_count()
    if cuda_device_cnt > 0:
        use_num_cuda_devices = 1
    else:
        use_num_cuda_devices = None

    trainer = pl.Trainer(fast_dev_run=True, max_steps=None,
                         gradient_clip_val=10,
                         weights_summary=None, gpus=use_num_cuda_devices,
                         show_progress_bar=True)

    result = trainer.fit(model)
    # verify training completed
    assert result == 1


def test_move_data_to_device_torchtext_include_length_false():
    """Test if batches created by torchtext with include_lengths=False raise an exception."""

    class DebugModel(pl.LightningModule):

        def __init__(self):
            super(DebugModel, self).__init__()

            # setup data loader generating batches with fields consisting of tensors
            self.debug_data_loader, self.text_field = _get_torchtext_data_iterator(include_lengths=False)

            self.learning_rate = 0.001

            pad_idx = self.text_field.vocab.stoi['<pad>']
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

            self.INPUT_DIM = len(self.text_field.vocab)
            self.ENC_EMB_DIM = 4  # keep it small for debugging
            self.embedding = torch.nn.Embedding(self.INPUT_DIM, self.ENC_EMB_DIM)

            self.hid_dim = 4
            self.rnn = torch.nn.GRU(self.ENC_EMB_DIM, self.hid_dim, 1, bidirectional=False)
            self.out = torch.nn.Linear(self.hid_dim, self.embedding.num_embeddings)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def forward(self, input_seq):
            embedded = self.embedding(input_seq)
            outputs, hidden = self.rnn(embedded)
            output = outputs.squeeze(0)
            prediction = self.out(output)
            return prediction

        def training_step(self, batch, batch_nb):
            """ Needed for testing data transfer. """

            target = batch.text
            output = self.forward(target)
            loss = self.criterion(output[:-1].view(-1, output.shape[2]), target[1:].view(-1))

            prefix = 'train'
            tensorboard_logs = {f'{prefix}_loss': loss.item()}

            result = {'loss': loss, 'log': tensorboard_logs}
            return result

        def train_dataloader(self):
            return self.debug_data_loader

    model = DebugModel()

    cuda_device_cnt = torch.cuda.device_count()
    if cuda_device_cnt > 0:
        use_num_cuda_devices = 1
    else:
        use_num_cuda_devices = None

    trainer = pl.Trainer(fast_dev_run=True, max_steps=None,
                         gradient_clip_val=10,
                         weights_summary=None, gpus=use_num_cuda_devices,
                         show_progress_bar=True)

    result = trainer.fit(model)
    # verify training completed
    assert result == 1


def test_batch_move_data_to_device_torchtext_include_lengths_false():
    cuda_device_cnt = torch.cuda.device_count()
    if cuda_device_cnt > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_iterator, _ = _get_torchtext_data_iterator(include_lengths=True)
    data_iter = iter(data_iterator)
    batch = next(data_iter)

    # this call should not throw an error
    batch_on_device = move_data_to_device(batch, device)
    # tensor with data
    assert(batch_on_device.text[0].device == device)
    # tensor with length of data
    assert(batch_on_device.text[1].device == device)


def test_batch_move_data_to_device_torchtext_include_lengths_true():
    cuda_device_cnt = torch.cuda.device_count()
    if cuda_device_cnt > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_iterator, _ = _get_torchtext_data_iterator(include_lengths=False)
    data_iter = iter(data_iterator)
    batch = next(data_iter)

    # this call should not throw an error
    batch_on_device = move_data_to_device(batch, device)
    # tensor with data
    assert(batch_on_device.text[0].device == device)
