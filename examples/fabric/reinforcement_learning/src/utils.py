import argparse
import os
from distutils.util import strtobool
from typing import Optional

import gymnasium as gym
import numpy as np
import torch


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
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )

    # Distributed arguments
    parser.add_argument(
        "--per-rank-num-envs", type=int, default=2, help="the number of parallel game environments for each rank"
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
    args = parser.parse_args()
    return args


def layer_init(layer: torch.nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0, ortho_init: bool = True):
    if ortho_init:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
