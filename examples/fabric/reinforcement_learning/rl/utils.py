import argparse
import math
import os
from distutils.util import strtobool
from typing import TYPE_CHECKING, Optional, Union

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from rl.agent import PPOAgent, PPOLightningAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default", help="the name of this experiment")

    # PyTorch arguments
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, GPU training will be used. "
        "This affects also the distributed backend used (NCCL (gpu) vs GLOO (cpu))",
    )
    parser.add_argument(
        "--player-on-gpu",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, player will run on GPU (used only by `train_fabric_decoupled.py` script). "
        "This affects also the distributed backend used (NCCL (gpu) vs GLOO (cpu))",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )

    # Distributed arguments
    parser.add_argument("--num-envs", type=int, default=2, help="the number of parallel game environments")
    parser.add_argument(
        "--share-data",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle sharing data between processes",
    )
    parser.add_argument("--per-rank-batch-size", type=int, default=64, help="the batch size for each rank")

    # Environment arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="the id of the environment")
    parser.add_argument(
        "--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout"
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )

    # PPO arguments
    parser.add_argument("--total-timesteps", type=int, default=2**16, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="the learning rate of the optimizer")
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation"
    )
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument(
        "--activation-function",
        type=str,
        default="relu",
        choices=["relu", "tanh"],
        help="The activation function of the model",
    )
    parser.add_argument(
        "--ortho-init",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles the orthogonal initialization of the model",
    )
    parser.add_argument(
        "--normalize-advantages",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    return parser.parse_args()


def layer_init(
    layer: torch.nn.Module,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
    ortho_init: bool = True,
):
    if ortho_init:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_annealing(optimizer: torch.optim.Optimizer, update: int, num_updates: int, initial_lr: float):
    frac = 1.0 - (update - 1.0) / num_updates
    lrnow = frac * initial_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lrnow


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: Optional[str] = None, prefix: str = ""):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0 and run_name is not None:
            env = gym.wrappers.RecordVideo(
                env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
            )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


@torch.no_grad()
def test(
    agent: Union["PPOLightningAgent", "PPOAgent"], device: torch.device, logger: SummaryWriter, args: argparse.Namespace
):
    env = make_env(args.env_id, args.seed, 0, args.capture_video, logger.log_dir, "test")()
    step = 0
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=device)
    while not done:
        # Act greedly through the environment
        action = agent.get_greedy_action(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=device)
        step += 1
    logger.add_scalar("Test/cumulative_reward", cumulative_rew, 0)
    env.close()
