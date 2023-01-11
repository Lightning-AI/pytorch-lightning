"""
MAML - Raw PyTorch implementation using the Learn2Learn library

Adapted from https://github.com/learnables/learn2learn/blob/master/examples/vision/distributed_maml.py
Original Author: Séb Arnold - learnables.net

Paper: https://arxiv.org/pdf/1703.03400.pdf

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
import numpy as np

import torch
import cherry
import learn2learn as l2l

WORLD_SIZE = 2


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    (support_data, support_labels), (query_data, query_labels) = l2l.data.partition_task(
        data=data,
        labels=labels,
        shots=shots,
    )

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(support_data), support_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(query_data)
    evaluation_error = loss(predictions, query_labels)
    evaluation_accuracy = l2l.utils.accuracy(predictions, query_labels)
    return evaluation_error, evaluation_accuracy


def main(
        ways=5,
        shots=5,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        rank=0,
        world_size=1,
        seed=42,
):
    print(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device_id = rank % torch.cuda.device_count()
        device = torch.device('cuda:' + str(device_id))
    print(rank, ':', device)

    # Create Tasksets using the benchmark interface
    # if rank == 0:
    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
                                                  train_ways=ways,
                                                  train_samples=2*shots,
                                                  test_ways=ways,
                                                  test_samples=2*shots,
                                                  num_tasks=20000,
                                                  root='~/data',
                                                  )
    # torch.distributed.barrier()

    # Create model
    # model = l2l.vision.models.MiniImagenetCNN(ways)
    model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = torch.optim.Adam(maml.parameters(), meta_lr)
    opt = cherry.optim.Distributed(maml.parameters(), opt=opt, sync=1)
    opt.sync_parameters()
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    print(rank, ':', device)

    for iteration in range(num_iterations):
        opt.zero_grad()
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
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()  # averages gradients across all workers

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
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


def distributed_main():
    local_rank = int(os.environ["LOCAL_RANK"])
    print("init2")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    torch.distributed.init_process_group(
        'gloo',
        # init_method='tcp://localhost:23456',
        rank=local_rank,
        world_size=WORLD_SIZE,
    )
    print("init3")

    rank = torch.distributed.get_rank()
    main(
        seed=42 + rank,
        rank=rank,
        world_size=WORLD_SIZE,
        meta_batch_size=32 // WORLD_SIZE,
    )


if __name__ == '__main__':
    distributed_main()