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
    torchrun --nproc_per_node=2 --standalone train_torch.py
"""

import argparse
import os
import random
import time
from datetime import datetime
from typing import Dict

import gymnasium as gym
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.optim as optim
from rl.agent import PPOAgent
from rl.loss import entropy_loss, policy_loss, value_loss
from rl.utils import linear_annealing, make_env, parse_args, test
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter


def train(
    agent: PPOAgent,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, Tensor],
    logger: SummaryWriter,
    global_step: int,
    args: argparse.Namespace,
):
    indexes = list(range(data["obs"].shape[0]))
    if args.share_data:
        sampler = DistributedSampler(
            indexes,
            num_replicas=distributed.get_world_size(),
            rank=distributed.get_rank(),
            shuffle=True,
            seed=args.seed,
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)
    per_epoch_losses = torch.tensor([0.0, 0.0, 0.0], device=data["obs"].device)
    for epoch in range(args.update_epochs):
        if args.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            _, newlogprob, entropy, newvalue = agent(data["obs"][batch_idxes], data["actions"].long()[batch_idxes])
            logratio = newlogprob - data["logprobs"][batch_idxes]
            ratio = logratio.exp()

            advantages = data["advantages"][batch_idxes]
            if args.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            pg_loss = policy_loss(advantages, ratio, args.clip_coef)
            per_epoch_losses[0] += pg_loss.detach()

            # Value loss
            v_loss = value_loss(
                newvalue,
                data["values"][batch_idxes],
                data["returns"][batch_idxes],
                args.clip_coef,
                args.clip_vloss,
                args.vf_coef,
            )
            per_epoch_losses[1] += v_loss.detach()

            # Entropy loss
            ent_loss = entropy_loss(entropy, args.ent_coef)
            per_epoch_losses[2] += ent_loss.detach()

            # Overall loss
            loss = pg_loss + ent_loss + v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        # Log
        distributed.reduce(per_epoch_losses, dst=0)
        if logger is not None:
            per_epoch_losses = per_epoch_losses / (len(sampler) * distributed.get_world_size())
            logger.add_scalar("Loss/policy_loss", per_epoch_losses[0], global_step)
            logger.add_scalar("Loss/value_loss", per_epoch_losses[1], global_step)
            logger.add_scalar("Loss/entropy_loss", per_epoch_losses[2], global_step)
        per_epoch_losses.fill_(0)


def main(args: argparse.Namespace):
    # Init distributed environment
    is_cuda_available = torch.cuda.is_available() and args.cuda
    backend = "nccl" if is_cuda_available else "gloo"
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if is_cuda_available:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if is_cuda_available else "cpu")
    distributed.init_process_group(backend=backend)

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Logger
    log_dir = None
    logger = None
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    if global_rank == 0:
        log_dir = os.path.join("logs", "torch_logs", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"), run_name)
        logger = SummaryWriter(log_dir=log_dir)

    # Log hyperparameters
    if global_rank == 0:
        logger.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # Environment setup
    envs = gym.vector.SyncVectorEnv([
        make_env(
            args.env_id,
            args.seed + global_rank * args.num_envs + i,
            global_rank,
            args.capture_video,
            logger.log_dir if global_rank == 0 else None,
            "train",
        )
        for i in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Define the agent and the optimizer and setup them with DistributedDataParallel
    agent: PPOAgent = PPOAgent(envs, act_fun=args.activation_function, ortho_init=args.ortho_init).to(device)
    agent = DistributedDataParallel(
        agent,
        device_ids=[local_rank] if is_cuda_available else None,
        output_device=local_rank if is_cuda_available else None,
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-4)

    # Local data
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    local_rew = 0.0
    local_ep_len = 0.0
    local_num_episodes = 0.0

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_step = int(args.num_envs * args.num_steps * world_size)
    num_updates = args.total_timesteps // single_global_step

    # Get the first environment observation and start the optimization
    next_obs = torch.tensor(envs.reset(seed=args.seed)[0], device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            linear_annealing(optimizer, update, num_updates, args.learning_rate)
        if global_rank == 0:
            logger.add_scalar("Info/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        for step in range(0, args.num_steps):
            global_step += args.num_envs * world_size
            obs[step] = next_obs
            dones[step] = next_done

            # Sample an action given the observation received by the environment
            with torch.no_grad():
                action, logprob, _, value = agent(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs, next_done = torch.tensor(next_obs, device=device), done.to(device)

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        if global_rank == 0:
                            print(
                                f"Rank-0: global_step={global_step}, "
                                f"reward_env_{i}={agent_final_info['episode']['r'][0]}"
                            )
                        local_num_episodes += 1
                        local_rew += agent_final_info["episode"]["r"][0]
                        local_ep_len += agent_final_info["episode"]["l"][0]

        # Sync the metrics
        global_stats = torch.tensor([local_rew, local_ep_len, local_num_episodes], device=device, dtype=torch.float32)
        distributed.reduce(global_stats, dst=0)
        if global_rank == 0 and global_stats[2] != 0.0:
            logger.add_scalar("Rewards/rew_avg", global_stats[0] / global_stats[2], global_step)
            logger.add_scalar("Game/ep_len_avg", global_stats[1] / global_stats[2], global_step)

        # Reset metrics
        local_rew = 0
        local_ep_len = 0
        local_num_episodes = 0

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        returns, advantages = agent.module.estimate_returns_and_advantages(
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

        if args.share_data:
            # Gather all the tensors from all the world, concat and reshape them
            gathered_data = [None for _ in range(world_size)]
            distributed.all_gather_object(gathered_data, local_data)
            processed_gathered_data = gathered_data[0]
            for i in range(1, len(gathered_data)):
                for k in processed_gathered_data:
                    processed_gathered_data[k] = torch.cat(
                        (processed_gathered_data[k].to(device), gathered_data[i][k].to(device)), dim=0
                    )
        else:
            processed_gathered_data = local_data

        # Train the agent
        train(agent, optimizer, processed_gathered_data, logger, global_step, args)
        if global_rank == 0:
            logger.add_scalar("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    if global_rank == 0:
        test(agent.module, device, logger, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
