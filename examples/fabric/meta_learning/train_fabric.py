"""
MAML - Accelerated with Lightning Fabric

Adapted from https://github.com/learnables/learn2learn/blob/master/examples/vision/distributed_maml.py
Original code author: Séb Arnold - learnables.net
Based on the paper: https://arxiv.org/abs/1703.03400

Requirements:
- lightning>=1.9.0
- learn2learn
- cherry-rl
- gym<=0.22

Run it with:
    fabric run train_fabric.py --accelerator=cuda --devices=2 --strategy=ddp
"""

import cherry
import learn2learn as l2l
import torch
from lightning.fabric import Fabric, seed_everything


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways):
    data, labels = batch

    # Separate data into adaptation/evalutation sets
    adaptation_indices = torch.zeros(data.size(0), dtype=bool)
    adaptation_indices[torch.arange(shots * ways) * 2] = True
    evaluation_indices = ~adaptation_indices
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
    ways=5,
    shots=5,
    meta_lr=0.003,
    fast_lr=0.5,
    meta_batch_size=32,
    adaptation_steps=1,
    num_iterations=60000,
    seed=42,
):
    # Create the Fabric object
    # Arguments get parsed from the command line, see `fabric run --help`
    fabric = Fabric()

    meta_batch_size = meta_batch_size // fabric.world_size
    seed_everything(seed + fabric.global_rank)

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(
        # 'mini-imagenet' works too, but you need to download it manually due to license restrictions of ImageNet
        "omniglot",
        train_ways=ways,
        train_samples=2 * shots,
        test_ways=ways,
        test_samples=2 * shots,
        num_tasks=20000,
        root="data",
    )

    # Create model
    # model = l2l.vision.models.MiniImagenetCNN(ways)
    model = l2l.vision.models.OmniglotFC(28**2, ways)
    model = fabric.to_device(model)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), meta_lr)
    optimizer = cherry.optim.Distributed(maml.parameters(), opt=optimizer, sync=1)

    # model, optimizer = fabric.setup(model, optimizer)

    optimizer.sync_parameters()
    loss = torch.nn.CrossEntropyLoss(reduction="mean")

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = fabric.to_device(tasksets.train.sample())
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                adaptation_steps,
                shots,
                ways,
            )
            fabric.backward(evaluation_error)
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = fabric.to_device(tasksets.validation.sample())
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                adaptation_steps,
                shots,
                ways,
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        fabric.print("\n")
        fabric.print("Iteration", iteration)
        fabric.print("Meta Train Error", meta_train_error / meta_batch_size)
        fabric.print("Meta Train Accuracy", meta_train_accuracy / meta_batch_size)
        fabric.print("Meta Valid Error", meta_valid_error / meta_batch_size)
        fabric.print("Meta Valid Accuracy", meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()  # averages gradients across all workers

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = fabric.to_device(tasksets.test.sample())
        evaluation_error, evaluation_accuracy = fast_adapt(
            batch,
            learner,
            loss,
            adaptation_steps,
            shots,
            ways,
        )
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    fabric.print("Meta Test Error", meta_test_error / meta_batch_size)
    fabric.print("Meta Test Accuracy", meta_test_accuracy / meta_batch_size)


if __name__ == "__main__":
    main()
