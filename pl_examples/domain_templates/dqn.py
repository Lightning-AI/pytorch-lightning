from typing import Tuple, OrderedDict, List

import pytorch_lightning as pl
import argparse
import gym
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


class DQN(nn.Module):
    """ Simple MLP network"""

    def __init__(self, obs_size, n_actions, hidden_size=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


# Named tuple for storing experience steps gathered in training
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class ExperienceBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them"""

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training
    """

    def __init__(self, buffer: ExperienceBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """
    Base Agent class handeling the interaction with the environment
    """

    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer) -> None:
        self.env = env
        self.exp_buffer = exp_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a)
            if device not in ['cpu']:
                state_v = state_v.cuda(device)

            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment"""

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)

        self.state = new_state
        if is_done:
            self.reset()
        return reward, is_done


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        self.env = gym.make(self.hparams.env)
        self.net = DQN(self.env.observation_space.shape,
                       self.env.action_space.n)
        self.tgt_net = DQN(self.env.observation_space.shape,
                           self.env.action_space.n)
        self.buffer = ExperienceBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets an action as an output
        """
        if isinstance(x, list):
            x = x[0]
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.tgt_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        return nn.MSELoss()(state_action_values,
                            expected_state_action_values)

    def training_step(self, batch, nb_batch) -> OrderedDict:
        """ Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved"""
        device = self.get_device(batch)
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                      self.global_step + 1 / self.hparams.eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.tgt_net.load_state_dict(self.net.state_dict())

        log = {'total_reward': torch.tensor(self.total_reward).to(device),
               'reward': torch.tensor(reward).to(device),
               'steps': torch.tensor(self.global_step).to(device)}

        return collections.OrderedDict({'loss': loss, 'log': log, 'progress_bar': log})

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                sampler=None
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        if self.on_gpu:
            device = batch[0].device.index
        else:
            device = 'cpu'

        return device


def main(hparams) -> None:
    model = DQNLightning(hparams)

    trainer = pl.Trainer(gpus=1, distributed_backend='dp', early_stop_callback=False, val_check_interval=100)

    trainer.fit(model)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=10,
                        help="how many frames do we update the target network")
    parser.add_argument("--replay_size", type=int, default=1000,
                        help="capacity of the replay buffer")
    parser.add_argument("--warm_start_size", type=int, default=1000,
                        help="how many samples do we use to fill our buffer at the start of training")
    parser.add_argument("--eps_last_frame", type=int, default=1000,
                        help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
    parser.add_argument("--max_episode_reward", type=int, default=200,
                        help="max episode reward in the environment")
    parser.add_argument("--warm_start_steps", type=int, default=1000,
                        help="max episode reward in the environment")

    args = parser.parse_args()

    main(args)
