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
    import torch.optim as optim
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule

    class LitModel(LightningModule):

        def __init__(self):
            super().__init__()

            # 1. Switch to manual optimization
            self.automatic_optimization = False

            self.truncated_bptt_steps = 10
            self.my_rnn = ParityModuleRNN() # Define RNN model using ParityModuleRNN

        # 2. Remove the `hiddens` argument
        def training_step(self, batch, batch_idx):

            # 3. Split the batch in chunks along the time dimension
            split_batches = split_batch(batch, self.truncated_bptt_steps)

            batch_size = 10
            hidden_dim = 20
            hiddens = torch.zeros(1, batch_size, hidden_dim, device=self.device)
            for split_batch in range(split_batches):
                # 4. Perform the optimization in a loop
                loss, hiddens = self.my_rnn(split_batch, hiddens)
                self.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 5. "Truncate"
                hiddens = hiddens.detach()

            # 6. Remove the return of `hiddens`
            # Returning loss in manual optimization is not needed
            return None

        def configure_optimizers(self):
            return optim.Adam(self.my_rnn.parameters(), lr=0.001)

    if __name__ == "__main__":
        model = LitModel()
        trainer = pl.Trainer(max_epochs=5)
        trainer.fit(model, train_dataloader) # Define your own dataloader
