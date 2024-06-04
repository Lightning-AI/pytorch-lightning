import lightning as L
import torch
import torch.nn.functional as F
from data import RandomTokenDataset
from lightning.fabric.strategies import ModelParallelStrategy
from model import ModelArgs, Transformer
from parallelism import parallelize
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import DataLoader


def train():
    strategy = ModelParallelStrategy(
        # User-defined function that applies the desired parallelizations specific to the model
        # (TP, FSDP2, activation checkpointing, ...)
        parallelize_fn=parallelize,
        # Define the size of the 2D parallelism
        # Set to "auto" to apply TP intra-node and DP inter-node
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    fabric = L.Fabric(accelerator="cuda", devices=4, strategy=strategy)
    fabric.launch()

    # Initialize the model
    model_args = ModelArgs(vocab_size=32000)
    with fabric.init_module(empty_init=True):
        model = Transformer(model_args)

    fabric.print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f} B")

    # Set up model and optimizer
    model = fabric.setup(model)
    model.init_weights()

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=True)
    optimizer = fabric.setup_optimizers(optimizer)

    # Define dataset/dataloader
    dataset = RandomTokenDataset(vocab_size=model_args.vocab_size, seq_length=128)
    dataloader = DataLoader(dataset, batch_size=8)

    # Fabric configures the sampler automatically for you such that
    # all batches in a tensor-parallel group are identical
    dataloader = fabric.setup_dataloaders(dataloader)

    # Simplified training loop
    fabric.print("Starting training ...")

    for i, batch in enumerate(dataloader):
        inputs = batch[:, :-1]
        labels = batch[:, 1:]

        output = model(inputs)

        with loss_parallel():
            loss = F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))
            fabric.backward(loss)

        optimizer.step()
        optimizer.zero_grad()
        fabric.print(f"Iteration {i} complete")

    # See `fabric consolidate --help` if you need to convert the checkpoint to a single file
    fabric.print("Saving a (distributed) checkpoint ...")
    state = {"model": model, "optimizer": optimizer, "iteration": i}
    fabric.save("checkpoint.pt", state)

    fabric.print("Training successfully completed!")
    fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


if __name__ == "__main__":
    assert torch.cuda.device_count() >= 4, "This example requires at least 4 GPUs with 24 GB of memory each."
    torch.set_float32_matmul_precision("high")
    train()
