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
    fabric run --devices=2 train_fabric_decoupled.py
"""

import argparse
import os
import time
from contextlib import nullcontext
from datetime import datetime

import gymnasium as gym
import torch
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from rl.agent import PPOLightningAgent
from rl.utils import linear_annealing, make_env, parse_args, test
from torch.distributed.algorithms.join import Join
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import MeanMetric


@torch.no_grad()
def player(args, world_collective: TorchCollective, player_trainer_collective: TorchCollective):
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}"
    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "fabric_decoupled_logs", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
        name=run_name,
    )
    log_dir = logger.log_dir

    # Initialize Fabric object
    fabric = Fabric(loggers=logger, accelerator="cuda" if args.player_on_gpu else "cpu")
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Log hyperparameters
    logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Environment setup
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed + i, 0, args.capture_video, log_dir, "train") for i in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Define the agent
    agent: PPOLightningAgent = PPOLightningAgent(
        envs,
        act_fun=args.activation_function,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ortho_init=args.ortho_init,
        normalize_advantages=args.normalize_advantages,
    ).to(device)
    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), device=device
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, agent.parameters())

    # Player metrics
    rew_avg = MeanMetric(sync_on_compute=False).to(device)
    ep_len_avg = MeanMetric(sync_on_compute=False).to(device)

    # Local data
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_step = int(args.num_envs * args.num_steps)
    num_updates = args.total_timesteps // single_global_step
    if not args.share_data:
        if single_global_step < world_collective.world_size - 1:
            raise RuntimeError(
                f"The number of trainers ({world_collective.world_size - 1})"
                f" is greater than the available collected data ({single_global_step})."
                f" Consider to lower the number of trainers at least to the size of available collected data"
            )
        chunks_sizes = [
            len(chunk)
            for chunk in torch.tensor_split(torch.arange(single_global_step), world_collective.world_size - 1)
        ]

    # Broadcast num_updates to all the world
    update_t = torch.tensor([num_updates], device=device, dtype=torch.float32)
    world_collective.broadcast(update_t, src=0)

    # Get the first environment observation and start the optimization
    next_obs = torch.tensor(envs.reset(seed=args.seed)[0], device=device)
    next_done = torch.zeros(args.num_envs).to(device)
    for _ in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Sample an action given the observation received by the environment
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))
            rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32).view(-1)
            next_obs, next_done = torch.tensor(next_obs, device=device), done.to(device)

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        rew_avg(agent_final_info["episode"]["r"][0])
                        ep_len_avg(agent_final_info["episode"]["l"][0])

        # Sync the metrics
        rew_avg_reduced = rew_avg.compute()
        if not rew_avg_reduced.isnan():
            fabric.log("Rewards/rew_avg", rew_avg_reduced, global_step)
        ep_len_avg_reduced = ep_len_avg.compute()
        if not ep_len_avg_reduced.isnan():
            fabric.log("Game/ep_len_avg", ep_len_avg_reduced, global_step)
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
        if not args.player_on_gpu and args.cuda:
            for v in local_data.values():
                v = v.pin_memory()

        # Send data to the training agents
        if args.share_data:
            world_collective.broadcast_object_list([local_data], src=0)
        else:
            # Split data in an even way, when possible
            perm = torch.randperm(single_global_step, device=device)
            chunks = [{} for _ in range(world_collective.world_size - 1)]
            for k, v in local_data.items():
                chunked_local_data = v[perm].split(chunks_sizes)
                for i in range(len(chunks)):
                    chunks[i][k] = chunked_local_data[i]

            world_collective.scatter_object_list([None], [None] + chunks, src=0)

        # Gather metrics from the trainers to be plotted
        metrics = [None]
        player_trainer_collective.broadcast_object_list(metrics, src=1)

        # Wait the trainers to finish
        player_trainer_collective.broadcast(flattened_parameters, src=1)

        # Convert back the parameters
        torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, agent.parameters())

        fabric.log_dict(metrics[0], global_step)
        fabric.log_dict({"Time/step_per_second": int(global_step / (time.time() - start_time))}, global_step)

    if args.share_data:
        world_collective.broadcast_object_list([-1], src=0)
    else:
        world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)
    envs.close()
    test(agent, device, fabric.logger.experiment, args)


def trainer(
    args,
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
):
    global_rank = world_collective.rank
    group_rank = global_rank - 1
    group_world_size = world_collective.world_size - 1

    # Initialize Fabric
    fabric = Fabric(strategy=DDPStrategy(process_group=optimization_pg), accelerator="cuda" if args.cuda else "cpu")
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, 0, False, None)])
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
        process_group=optimization_pg,
    )
    optimizer = agent.configure_optimizers(args.learning_rate)
    agent, optimizer = fabric.setup(agent, optimizer)

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), src=1
        )

    # Receive maximum number of updates from the player
    update = 0
    num_updates = torch.zeros(1, device=device)
    world_collective.broadcast(num_updates, src=0)
    num_updates = num_updates.item()

    # Start training
    while True:
        # Wait for data
        data = [None]
        if args.share_data:
            world_collective.broadcast_object_list(data, src=0)
        else:
            world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
        data = data[0]
        if data == -1:
            return

        # Metrics dict to be sent to the player
        if group_rank == 0:
            metrics = {}

        # Lerning rate annealing
        if args.anneal_lr:
            linear_annealing(optimizer, update, num_updates, args.learning_rate)
        if group_rank == 0:
            metrics["Info/learning_rate"] = optimizer.param_groups[0]["lr"]
        update += 1

        indexes = list(range(data["obs"].shape[0]))
        if args.share_data:
            sampler = DistributedSampler(
                indexes, num_replicas=group_world_size, rank=group_rank, shuffle=True, seed=args.seed, drop_last=False
            )
        else:
            sampler = RandomSampler(indexes)
        sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)

        # The Join context is needed because there can be the possibility
        # that some ranks receive less data
        with Join([agent._forward_module]) if not args.share_data else nullcontext():
            for epoch in range(args.update_epochs):
                if args.share_data:
                    sampler.sampler.set_epoch(epoch)
                for batch_idxes in sampler:
                    loss = agent.training_step({k: v[batch_idxes].to(device) for k, v in data.items()})
                    optimizer.zero_grad(set_to_none=True)
                    fabric.backward(loss)
                    fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
                    optimizer.step()

        # Sync metrics
        avg_pg_loss = agent.avg_pg_loss.compute()
        avg_value_loss = agent.avg_value_loss.compute()
        avg_ent_loss = agent.avg_ent_loss.compute()
        agent.reset_metrics()

        # Send updated weights to the player
        if global_rank == 1:
            metrics["Loss/policy_loss"] = avg_pg_loss
            metrics["Loss/value_loss"] = avg_value_loss
            metrics["Loss/entropy_loss"] = avg_ent_loss
            player_trainer_collective.broadcast_object_list(
                [metrics], src=1
            )  # Broadcast metrics: fake send with object list between rank-0 and rank-1
            player_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), src=1
            )


def main(args: argparse.Namespace):
    world_collective = TorchCollective()
    player_trainer_collective = TorchCollective()
    world_collective.setup(backend="nccl" if args.player_on_gpu and args.cuda else "gloo")

    # Create a global group, assigning it to the collective: used by the player to exchange
    # collected experiences with the trainers
    world_collective.create_group()
    global_rank = world_collective.rank

    # Create a group between rank-0 (player) and rank-1 (trainer), assigning it to the collective:
    # used by rank-1 to send metrics to be tracked by the rank-0 at the end of a training episode
    player_trainer_collective.create_group(ranks=[0, 1])

    # Create a new group, without assigning it to the collective: in this way the trainers can
    # still communicate with the player through the global group, but they can optimize the agent
    # between themselves
    optimization_pg = world_collective.new_group(ranks=list(range(1, world_collective.world_size)))
    if global_rank == 0:
        player(args, world_collective, player_trainer_collective)
    else:
        trainer(args, world_collective, player_trainer_collective, optimization_pg)


if __name__ == "__main__":
    args = parse_args()
    main(args)
