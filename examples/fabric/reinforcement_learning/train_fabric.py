"""
Proximal Policy Optimization (PPO) - Accelerated with Lightning Fabric

Author: Federico Belotti @belerico
Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
Based on the paper: https://arxiv.org/abs/1707.06347

Requirements:
- gymnasium[box2d]>=0.27.1
- moviepy
- lightning
- torchmetrics
- tensorboard


Run it with:
    lightning run model --accelerator=cpu --strategy=ddp --devices=2 train_fabric.py
"""

import argparse
import os
import time
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from src.loss import entropy_loss, policy_loss, value_loss
from src.utils import layer_init, make_env, parse_args
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, DistributedSampler
from torchmetrics import MeanMetric

from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.pytorch import LightningModule


class PPOLightningAgent(LightningModule):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        act_fun: str = "relu",
        ortho_init: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        normalize_advantages: bool = False,
    ):
        super().__init__()
        if act_fun.lower() == "relu":
            act_fun = torch.nn.ReLU()
        elif act_fun.lower() == "tanh":
            act_fun = torch.nn.Tanh()
        else:
            raise ValueError("Unrecognized activation function: `act_fun` must be either `relu` or `tanh`")
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages
        self.critic = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64), ortho_init=ortho_init
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, 1), std=1.0, ortho_init=ortho_init),
        )
        self.actor = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64), ortho_init=ortho_init
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, envs.single_action_space.n), std=0.01, ortho_init=ortho_init),
        )
        self.avg_pg_loss = MeanMetric()
        self.avg_value_loss = MeanMetric()
        self.avg_ent_loss = MeanMetric()

    def get_action(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()

    def get_greedy_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy = self.get_action(x, action)
        value = self.get_value(x)
        return action, log_prob, entropy, value

    def forward(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_action_and_value(x, action)

    @torch.no_grad()
    def estimate_returns_and_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_value = self.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

    def training_step(self, batch: Dict[str, torch.Tensor]):
        # Get actions and values given the current observations
        _, newlogprob, entropy, newvalue = self(batch["obs"], batch["actions"].long())
        logratio = newlogprob - batch["logprobs"]
        ratio = logratio.exp()

        # Policy loss
        advantages = batch["advantages"]
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = policy_loss(batch["advantages"], ratio, self.clip_coef)

        # Value loss
        v_loss = value_loss(
            newvalue,
            batch["values"],
            batch["returns"],
            self.clip_coef,
            self.clip_vloss,
            self.vf_coef,
        )

        # Entropy loss
        ent_loss = entropy_loss(entropy, self.ent_coef)

        # Update metrics
        self.avg_pg_loss(pg_loss)
        self.avg_value_loss(v_loss)
        self.avg_ent_loss(ent_loss)

        # Overall loss
        return pg_loss + ent_loss + v_loss

    def on_train_epoch_end(self, global_step: int) -> None:
        # Log metrics and reset their internal state
        self.logger.log_metrics(
            {
                "Loss/policy_loss": self.avg_pg_loss.compute(),
                "Loss/value_loss": self.avg_value_loss.compute(),
                "Loss/entropy_loss": self.avg_ent_loss.compute(),
            },
            global_step,
        )
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)


def train(
    fabric: Fabric,
    agent: PPOLightningAgent,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, torch.Tensor],
    global_step: int,
    args: argparse.Namespace,
):
    sampler = DistributedSampler(
        list(range(data["obs"].shape[0])),
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=True,
        seed=args.seed,
    )
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)

    for epoch in range(args.update_epochs):
        sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            loss = agent.training_step({k: v[batch_idxes] for k, v in data.items()})
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
        agent.on_train_epoch_end(global_step)


@torch.no_grad()
def test(fabric: Fabric, agent: PPOLightningAgent, logger: TensorBoardLogger, args: argparse.Namespace):
    device = fabric.device
    env = make_env(
        args.env_id, args.seed + fabric.global_rank, fabric.global_rank, args.capture_video, logger.log_dir, "test"
    )()
    step = 0
    done = False
    cumulative_rew = 0
    next_obs = torch.Tensor(env.reset(seed=args.seed)[0]).to(device)
    while not done:
        # Act greedly through the environment
        action = agent.get_greedy_action(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
        done = np.logical_or(done, truncated)
        cumulative_rew += reward
        next_obs = torch.Tensor(next_obs).to(device)
        step += 1
    fabric.log("Test/cumulative_reward", cumulative_rew, 0)
    env.close()


def main(args: argparse.Namespace):
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logger = TensorBoardLogger(root_dir=os.path.join("logs", "fabric_logs"), name=run_name)

    # Initialize Fabric
    fabric = Fabric(loggers=logger)
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Log hyperparameters
    fabric.logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + rank, rank, args.capture_video, logger.log_dir, "train")
            for _ in range(args.per_rank_num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    agent: PPOLightningAgent = PPOLightningAgent(
        envs,
        act_fun=args.activation_function,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ortho_init=args.ortho_init,
        normalize_advantages=args.normalize_advantages,
    )
    optimizer = agent.configure_optimizers(args.learning_rate)
    agent, optimizer = fabric.setup(agent, optimizer)

    # Player metrics
    rew_avg = torchmetrics.MeanMetric().to(device)
    ep_len_avg = torchmetrics.MeanMetric().to(device)

    # Local data
    obs = torch.zeros((args.num_steps, args.per_rank_num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.per_rank_num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.per_rank_num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.per_rank_num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.per_rank_num_envs), device=device)
    values = torch.zeros((args.num_steps, args.per_rank_num_envs), device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_rollout = int(args.per_rank_num_envs * args.num_steps * world_size)
    num_updates = args.total_timesteps // single_global_rollout

    # Get the first environment observation and start the optimization
    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
    next_done = torch.zeros(args.per_rank_num_envs, device=device)
    for update in range(1, num_updates + 1):
        # Learning rate annealing
        fabric.log("Info/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.per_rank_num_envs * world_size
            obs[step] = next_obs
            dones[step] = next_done

            # Sample an action given the observation received by the environment
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(done, truncated)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in info:
                for agent_id, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        if agent_id == 0:
                            fabric.print(
                                f"global_step={global_step}, reward_agent_0={agent_final_info['episode']['r'][0]}"
                            )
                        rew_avg(agent_final_info["episode"]["r"][0])
                        ep_len_avg(agent_final_info["episode"]["l"][0])

        # Sync the metrics
        fabric.log("Rewards/rew_avg", rew_avg.compute(), global_step)
        fabric.log("Game/ep_len_avg", ep_len_avg.compute(), global_step)
        rew_avg.reset()
        ep_len_avg.reset()

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        returns, advantages = agent.estimate_returns_and_advantages(
            rewards, values, dones, next_obs, next_done, args.num_steps, args.gamma, args.gae_lambda
        )

        # Flatten the batch
        local_data = {
            "obs": obs.reshape((-1,) + envs.single_observation_space.shape),
            "logprobs": logprobs.reshape(-1),
            "actions": actions.reshape((-1,) + envs.single_action_space.shape),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": values.reshape(-1),
        }

        # Gather all the tensors from all the world and reshape them
        gathered_data = fabric.all_gather(local_data)
        for k, v in gathered_data.items():
            if k == "obs":
                gathered_data[k] = v.reshape((-1,) + envs.single_observation_space.shape)
            elif k == "actions":
                gathered_data[k] = v.reshape((-1,) + envs.single_action_space.shape)
            else:
                gathered_data[k] = v.reshape(-1)

        # Train the agent
        train(fabric, agent, optimizer, gathered_data, global_step, args)
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    if fabric.is_global_zero:
        test(fabric, agent, logger, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
