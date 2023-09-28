import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.demos import Transformer, WikiText2
from torch.utils.data import DataLoader, random_split


def main():
    L.seed_everything(42)

    fabric = L.Fabric()

    # Data
    dataset = WikiText2()
    train_dataloader, val_dataloader, _ = get_dataloaders(dataset)

    # Model
    model = Transformer(vocab_size=dataset.vocab_size)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    train(fabric, model, optimizer, train_dataloader, val_dataloader)


def train(fabric, model, optimizer, train_dataloader, val_dataloader, max_epochs=20):
    for epoch in range(max_epochs):
        train_epoch(fabric, model, optimizer, train_dataloader, epoch)
        val_loss = validate(fabric, model, val_dataloader)
        fabric.print(f"val loss {val_loss.item():.4f}")


def train_epoch(fabric, model, optimizer, train_dataloader, epoch):
    for batch_idx, batch in enumerate(train_dataloader):
        input, target = batch
        output = model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, clip_val=0.25)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 200 == 0:
            fabric.print(f"epoch: {epoch} - iteration: {batch_idx} - loss {loss.item():.4f}")


@torch.no_grad()
def validate(fabric, model, val_dataloader):
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(len(val_dataloader))
    for k, batch in enumerate(val_dataloader):
        input, target = batch
        output = model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def get_dataloaders(dataset):
    n = len(dataset)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n - 4000, 2000, 2000], generator=generator)
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    main()
