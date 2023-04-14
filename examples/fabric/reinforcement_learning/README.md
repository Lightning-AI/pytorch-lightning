# Proximal Policy Optimization - PPO implementation powered by Lightning Fabric

This is an example of a Reinforcement Learning algorithm called [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) implemented in PyTorch and accelerated by [Lightning Fabric](https://lightning.ai/docs/fabric).

The goal of Reinforcement Learning is to train agents to act in their surrounding environment maximizing the cumulative reward received from it. This can be depicted in the following figure:

<p align="center">
  <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/reinforcement.png">
</p>

PPO is one of such algorithms, which alternates between sampling data through interaction with the environment, and optimizing a
“surrogate” objective function using stochastic gradient ascent.

## Requirements

Install requirements by running

```bash
pip install -r requirements.txt
```

## Example 1 - Environment coupled with the agent

In this example we present two code versions: the first one is implemented in raw PyTorch, but it contains quite a bit of boilerplate code for distributed training. The second one is using Lightning Fabric to accelerate and scale the model.

The main architecture is the following:

<p align="center">
  <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/fabric_coupled.png">
</p>

where `N+1` processes (labelled *rank-0*, ..., *rank-N* in the figure above) will be spawned by Fabric/PyTorch, each of them running `M+1` independent copies of the environment (*Env-0*, ..., *Env-M*). Every rank has its own copy of the agent, represented by a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)/[Pytorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), which will be updated through distributed training.

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
  --num-envs 2 \
  --num-steps 2048 \
  --seed 1
```

## Example 2 - Environment decoupled from the agent

In this example we have gone even further leveraging the flexibility offered by [Fabric](https://lightning.ai/docs/fabric).
The architecture is depicted in the following figure:

<p align="center">
  <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/ppo_fabric_decoupled.png">
</p>

where, differently from the previous example, we have completely decoupled the environment from the agents: the *rank-0* process will be regarded as the *Player*, which runs `M+1` independent copies of the environment (*Env-0*, ..., *Env-M*); the *rank-1*, ..., *rank-N* are the *Trainers*, which contain the agent to be optimized. Player and Trainer share data through [collectives](https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.plugins.collectives.TorchCollective.html#lightning.fabric.plugins.collectives.TorchCollective) and thanks to Fabric's flexibility we can run Player and Trainers on different devices.
So for example:

```bash
lightning run model --devices=3 train_fabric_decoupled.py --num-envs 4
```

will spawn 3 processes, one is the Player and the others the Trainers, with the Player running 4 independent environments, where every process runs on the CPU;

```bash
lightning run model --devices=3 train_fabric_decoupled.py --num-envs 4 --cuda
```

will instead run only the Trainers on the GPU.
If one wants to run both the Player and the Trainers on the GPU, then both the flags `--cuda` and `--player-on-gpu` must be provided:

```bash
lightning run model --devices=3 train_fabric_decoupled.py --num-envs 4 --cuda --player-on-gpu
```

> **Warning**
>
> With this second example, there is no need for the user to provide the `accellerator` and the `strategy` to the `lightning run model` script.
