# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deep Reinforcement Learning: Deep Q-network (DQN)

The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.

To run the template, just run:
`python reinforce_learn_Qnet.py`

After ~1500 steps, you will see the total_reward hitting the max score of 475+.
Open up TensorBoard to see the metrics:

`tensorboard --logdir default`

References
----------

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py

"""

import argparse
import random
from collections import OrderedDict, deque, namedtuple
from collections.abc import Iterator

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from lightning.pytorch import LightningModule, Trainer, cli_lightning_logo, seed_everything


class DQN(nn.Module):
    """Simple MLP network.

    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )

    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x.float())


# Named tuple for storing experience steps gathered in training
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "new_state"])


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    >>> ReplayBuffer(5)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.ReplayBuffer object at ...>

    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)

        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple:
        indices = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            torch.tensor(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
            torch.tensor(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    >>> RLDataset(ReplayBuffer(5))  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.RLDataset object at ...>

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """Base Agent class handling the interaction with the environment.

    >>> env = gym.make("CartPole-v1")
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.Agent object at ...>

    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action

        """
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu") -> tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done

        """
        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(LightningModule):
    """Basic DQN Model.

    >>> DQNLightning(env="CartPole-v1")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )

    """

    def __init__(
        self,
        env: str,
        replay_size: int = 200,
        warm_start_steps: int = 200,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_last_frame: int = 200,
        sync_rate: int = 10,
        lr: float = 1e-2,
        episode_length: int = 50,
        batch_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_length = episode_length
        self.batch_size = batch_size

        self.env = gym.make(env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state `x` through the network and gets the `q_values` of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        return self.net(x)

    def dqn_mse_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss

        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss based on
        the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """
        device = self.get_device(batch)
        epsilon = max(self.eps_end, self.eps_start - (self.global_step + 1) / self.eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "steps": torch.tensor(self.global_step).to(device),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

    def configure_optimizers(self) -> list[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.episode_length)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


def main(args) -> None:
    model = DQNLightning(**vars(args))
    trainer = Trainer(accelerator="cpu", devices=1, val_check_interval=100)
    trainer.fit(model)


if __name__ == "__main__":
    cli_lightning_logo()
    seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment tag")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
    parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
    parser.add_argument(
        "--warm_start_steps",
        type=int,
        default=1000,
        help="how many samples do we use to fill our buffer at the start of training",
    )
    parser.add_argument("--eps_last_frame", type=int, default=1000, help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
    args = parser.parse_args()

    main(args)
