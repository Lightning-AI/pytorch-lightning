##############################################
Truncated Backpropagation Through Time (TBPTT)
##############################################

Truncated Backpropagation Through Time (TBPTT) performs backpropogation every k steps of
a much longer sequence. This is made possible by passing training batches
split along the time-dimensions into splits of size k to the
``training_step``. In order to keep the same forward propagation behavior, all
hidden states should be kept in-between each time-dimension split.


.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    import lightning as L


    class AverageDataset(Dataset):
        def __init__(self, dataset_len=300, sequence_len=100):
            self.dataset_len = dataset_len
            self.sequence_len = sequence_len
            self.input_seq = torch.randn(dataset_len, sequence_len, 10)
            top, bottom = self.input_seq.chunk(2, -1)
            self.output_seq = top + bottom.roll(shifts=1, dims=-1)

        def __len__(self):
            return self.dataset_len

        def __getitem__(self, item):
            return self.input_seq[item], self.output_seq[item]


    class LitModel(L.LightningModule):

        def __init__(self):
            super().__init__()

            self.batch_size = 10
            self.in_features = 10
            self.out_features = 5
            self.hidden_dim = 20

            # 1. Switch to manual optimization
            self.automatic_optimization = False
            self.truncated_bptt_steps = 10

            self.rnn = nn.LSTM(self.in_features, self.hidden_dim, batch_first=True)
            self.linear_out = nn.Linear(in_features=self.hidden_dim, out_features=self.out_features)

        def forward(self, x, hs):
            seq, hs = self.rnn(x, hs)
            return self.linear_out(seq), hs

        # 2. Remove the `hiddens` argument
        def training_step(self, batch, batch_idx):
            # 3. Split the batch in chunks along the time dimension
            x, y = batch
            split_x, split_y = [
                x.tensor_split(self.truncated_bptt_steps, dim=1),
                y.tensor_split(self.truncated_bptt_steps, dim=1)
            ]

            hiddens = None
            optimizer = self.optimizers()
            losses = []

            # 4. Perform the optimization in a loop
            for x, y in zip(split_x, split_y):
                y_pred, hiddens = self(x, hiddens)
                loss = F.mse_loss(y_pred, y)

                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

                # 5. "Truncate"
                hiddens = [h.detach() for h in hiddens]
                losses.append(loss.detach())

            avg_loss = sum(losses) / len(losses)
            self.log("train_loss", avg_loss, prog_bar=True)

            # 6. Remove the return of `hiddens`
            # Returning loss in manual optimization is not needed
            return None

        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=0.001)

        def train_dataloader(self):
            return DataLoader(AverageDataset(), batch_size=self.batch_size)


    if __name__ == "__main__":
        model = LitModel()
        trainer = L.Trainer(max_epochs=5)
        trainer.fit(model)
