from collections import deque, namedtuple
import random

from pettingzoo.butterfly import knights_archers_zombies_v10
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_s):
        return random.sample(self.memory, batch_s)

    def __len__(self):
        return len(self.memory)


# initialize Deep Q learning network as class
class DQN(nn.Module):
    def __init__(self, n_dimensions, n_actions):
        super(DQN, self).__init__()
        flatten = n_dimensions[0] * n_dimensions[1]
        self.layer1 = nn.Linear(flatten, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.view(-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# get the number of actions for each agent
num_agent_act = np.array([6, 6, 6, 6])


class AgentDQN:
    def __init__(self, learning_rate, gamma, eps_start_val, eps_end_val, eps_decay, num_actions, observation_size):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_start_val = eps_start_val
        self.eps_decay = eps_decay
        self.eps_end_val = eps_end_val
        self.model = DQN(observation_size, num_actions)
        self.memory = ReplayBuffer(1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.steps_done = 0

    # if random returns random values between 0-5
    def select_action(self, s):

        sample = random.random()
        eps_threshold = self.eps_end_val + (self.eps_start_val - self.eps_end_val) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # return random.randrange(0, 6)

                return self.model(s).max(0).indices.item()
        else:
            return random.randrange(0, 6)

    def remember(self, s, a, r, new_s, d):
        self.memory.push(s, a, r, new_s, d)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = reward + self.gamma * torch.max(
                    self.model(next_state))
                target_f = self.model(state).detach().numpy()
                target_f[action] = target
                self.optimizer.zero_grad()
                # print(self.model(state))
                loss = nn.MSELoss()(torch.tensor(target_f), self.model(state))
                loss.backward()
                self.optimizer.step()
            if self.eps_start_val > self.eps_end_val:
                self.eps_start_val *= self.eps_decay


# define vectorized observation space where n is num_archers + num_knights + num_swords + max_arrows + max_zombies + 1
observation_shape = (27, 5)

all_agents = ['archer_0', 'archer_1', 'knight_0', 'knight_1']

archer_1 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[0], observation_size=observation_shape)

archer_2 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[1], observation_size=observation_shape)

knight_1 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[2], observation_size=observation_shape)

knight_2 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[3], observation_size=observation_shape)

env = knights_archers_zombies_v10.env(render_mode="human", spawn_rate=20, num_archers=2, num_knights=2, max_zombies=10,
                                      max_arrows=10, killable_knights=True, killable_archers=True, pad_observation=True,
                                      line_death=True, max_cycles=900,
                                      vector_state=True, use_typemasks=False, sequence_space=False)

num_episodes = 10
agent_list = [archer_1, archer_2, knight_1, knight_2]
batch_size = 64
last_observations = {agent: None for agent in agent_list}


total_reward_vals = np.array([], dtype=int)
for ep in range(num_episodes):
    env.reset(seed=42)
    total_reward = 0
    done = False
    while not done:
        for agent in env.agent_iter():
            if agent == 'archer_0':
                agent = agent_list[0]
            elif agent == 'archer_1':
                agent = agent_list[1]
            elif agent == 'knight_0':
                agent = agent_list[2]
            else:
                agent = agent_list[3]

            observation, reward, termination, truncation, info = env.last()
            done = truncation or termination
            if done:
                action = None

            else:
                action = agent.select_action(observation)

            env.step(action)

            # initially the observation step is the current state but after the next iteration
            # it becomes the next state for the replay buffer and the previous state becomes the current
            if last_observations[agent] is not None:
                agent.remember(last_observations[agent], action, reward, observation, termination)
                total_reward += int(reward)
                agent.replay(batch_size)

            last_observations[agent] = observation

        if done:
            env.reset()
            break

    total_reward_vals = np.append(total_reward_vals, total_reward)
    print(total_reward_vals)
    print(f"Episode {ep + 1} finished with total reward: {total_reward}")

env.close()
# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#
#
#
#
#         # this is where you would insert your policy
#         # action = env.action_space(agent).sample()
#
#     env.step(action)
# env.close()

# episode_duration = []
# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())
#

# policy_network_a1 = DQN(observation_shape, num_agent_act[0])
# policy_network_a2 = DQN(observation_shape, num_agent_act[1])
# policy_network_k1 = DQN(observation_shape, num_agent_act[2])
# policy_network_k2 = DQN(observation_shape, num_agent_act[3])
#
# optimizer = optim.AdamW(policy_network_a1.parameters(), lr=learning_rate, amsgrad=True)
# memory = ReplayBuffer(10000)
