# Proximal Policy Optimization - PPO implementation powered by Lightning Fabric

This is an example of a Reinforcement Learning algorithm called [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) implemented in PyTorch and accelerated by [Lightning Fabric](https://pytorch-lightning.readthedocs.io/en/latest/fabric/fabric.html).

The goal of Reinforcement Learning is to train agents to act in their surrounding environment maximizing the cumulative reward received from it. This can be depicted in the following figure:

<p align="center">
  <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/reinforcement.png">
</p>

PPO is one of such algorithms, which alternates between sampling data through interaction with the environment, and optimizing a
“surrogate” objective function using stochastic gradient ascent.

We present two code versions: the first one is implemented in raw PyTorch, but it contains quite a bit of boilerplate code for distributed training. The second one is using Lightning Fabric to accelerate and scale the model.

Tip: You can easily inspect the difference between the two files with:

```
sdiff train_torch.py train_fabric.py
```

## Requirements

Install requirements by running

```
pip install -r requirements.txt
```

## Run

### Raw PyTorch:

```
torchrun --nproc_per_node=2 --standalone train_torch.py
```

You can visualize training and test logs by running

```
tensorboard --logdir torch_logs
```

### Lightning Fabric:

```
lightning run model --accelerator=cpu --strategy=ddp --devices=2 train_fabric.py
```

You can visualize training and test logs by running

```
tensorboard --logdir fabric_logs
```
