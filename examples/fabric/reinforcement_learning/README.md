# Proximal Policy Optimization - PPO implementation powered by Lightning Fabric

This is an example of a Reinforcement Learning algorithm called [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) implemented in PyTorch and accelerated by [Lightning Fabric](https://pytorch-lightning.readthedocs.io/en/stable/fabric/fabric.html).

The goal of Reinforcement Learning is to train agents to act in their surrounding environment maximizing the cumulative reward received from it. This can be depicted in the following figure:

<p align="center">
  <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/reinforcement.png">
</p>

PPO is one of such algorithms, which alternates between sampling data through interaction with the environment, and optimizing a
“surrogate” objective function using stochastic gradient ascent.

We present two code versions: the first one is implemented in raw PyTorch, but it contains quite a bit of boilerplate code for distributed training. The second one is using Lightning Fabric to accelerate and scale the model.

Tip: You can easily inspect the difference between the two files with:

```bash
sdiff train_torch.py train_fabric.py
```

## Requirements

Install requirements by running

```bash
pip install -r requirements.txt
```

## Run

### Raw PyTorch:

```bash
torchrun --nproc_per_node=2 --standalone train_torch.py
```

### Lightning Fabric:

```bash
lightning run model --accelerator=cpu --strategy=ddp --devices=2 train_fabric.py
```

### Visualizing logs

You can visualize training and test logs by running:

```bash
tensorboard --logdir logs
```

Under the `logs` folder you should find two folders:

- `logs/torch_logs`
- `logs/fabric_logs`

If you have run the experiment with the `--capture-video` you should find the `train_videos` and `test_videos` folders under the specific experiment folder.

## Results

The following video shows a trained agent on the [LunarLander-v2 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

<p align="center">
  <video controls>
    <source src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/test.mp4" type="video/mp4">
  </video>
</p>

The agent was trained with the following:

```bash
lightning run model \
  --accelerator=cpu \
  --strategy=ddp \
  --devices=2 \
  train_fabric.py \
  --capture-video \
  --env-id LunarLander-v2 \
  --total-timesteps 500000 \
  --ortho-init \
  --per-rank-num-envs 2 \
  --num-steps 2048 \
  --seed 1
```
