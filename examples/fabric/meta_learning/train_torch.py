"""
MAML - Raw PyTorch implementation using the Learn2Learn library

Adapted from https://github.com/learnables/learn2learn/blob/master/examples/vision/distributed_maml.py
Original code author: Séb Arnold - learnables.net
Based on the paper: https://arxiv.org/abs/1703.03400

Requirements:
- learn2learn
- cherry-rl
- gym<=0.22

This code is written for distributed training.

Run it with:
    torchrun --nproc_per_node=2 --standalone train_torch.py
"""

import os
import random

import cherry
import learn2learn as l2l
import torch
import torch.distributed as dist


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

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
    cuda=True,
    seed=42,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("gloo", rank=local_rank, world_size=world_size)
    rank = dist.get_rank()

    meta_batch_size = meta_batch_size // world_size
    seed = seed + rank

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device_id = rank % torch.cuda.device_count()
        device = torch.device("cuda:" + str(device_id))

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
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), meta_lr)
    optimizer = cherry.optim.Distributed(maml.parameters(), opt=optimizer, sync=1)
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
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                adaptation_steps,
                shots,
                ways,
                device,
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                adaptation_steps,
                shots,
                ways,
                device,
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        if rank == 0:
            print("\n")
            print("Iteration", iteration)
            print("Meta Train Error", meta_train_error / meta_batch_size)
            print("Meta Train Accuracy", meta_train_accuracy / meta_batch_size)
            print("Meta Valid Error", meta_valid_error / meta_batch_size)
            print("Meta Valid Accuracy", meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()  # averages gradients across all workers

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(
            batch,
            learner,
            loss,
            adaptation_steps,
            shots,
            ways,
            device,
        )
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print("Meta Test Error", meta_test_error / meta_batch_size)
    print("Meta Test Accuracy", meta_test_accuracy / meta_batch_size)


if __name__ == "__main__":
    main()
