import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PGN(nn.Module):
    def __init__(self):
        super(PGN, self).__init__()
        self.linear1 = nn.Linear(4, 24)
        self.linear2 = nn.Linear(24, 36)
        self.linear3 = nn.Linear(36, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


class CartAgent(object):
    def __init__(self, learning_rate, gamma):
        self.pgn = PGN()
        self.gamma = gamma

        self._init_memory()
        self.optimizer = torch.optim.RMSprop(self.pgn.parameters(), lr=learning_rate)

    def memorize(self, state, action, reward):
        # save to memory for mini-batch gradient descent
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)
        self.steps += 1

    def learn(self):
        self._adjust_reward()

        # policy gradient
        self.optimizer.zero_grad()
        for i in range(self.steps):
            # all steps in multi games
            state = self.state_pool[i]
            action = torch.FloatTensor([self.action_pool[i]])
            reward = self.reward_pool[i]

            probs = self.act(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward
            loss.backward()
        self.optimizer.step()

        self._init_memory()

    def act(self, state):
        return self.pgn(state)

    def _init_memory(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0

    def _adjust_reward(self):
        # backward weight
        running_add = 0
        for i in reversed(range(self.steps)):
            if self.reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + self.reward_pool[i]
                self.reward_pool[i] = running_add

        # normalize reward
        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)
        for i in range(self.steps):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std


def train():
    # hyper parameter
    BATCH_SIZE = 5
    LEARNING_RATE = 0.01
    GAMMA = 0.99
    NUM_EPISODES = 500

    env = gym.make('CartPole-v1')
    cart_agent = CartAgent(learning_rate=LEARNING_RATE, gamma=GAMMA)

    for i_episode in range(NUM_EPISODES):
        next_state = env.reset()
        env.render(mode='rgb_array')

        for t in count():
            state = torch.from_numpy(next_state).float()

            probs = cart_agent.act(state)
            m = Bernoulli(probs)
            action = m.sample()

            action = action.data.numpy().astype(int).item()
            next_state, reward, done, _ = env.step(action)
            env.render(mode='rgb_array')

            # end action's reward equals 0
            if done:
                reward = 0

            cart_agent.memorize(state, action, reward)

            if done:
                logger.info({'Episode {}: durations {}'.format(i_episode, t)})
                break

        # update parameter every batch size
        if i_episode > 0 and i_episode % BATCH_SIZE == 0:
            cart_agent.learn()


if __name__ == '__main__':
    train()