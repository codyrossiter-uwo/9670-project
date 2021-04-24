import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from json import JSONDecodeError

import numpy as np
import random
from collections import defaultdict
from torch.distributions import Categorical

from agents.agent import Agent
from helper import state_to_string

class Actor(nn.Module):
    def __init__(self, action_space):
        super(Actor, self).__init__()
        #self.state_size = env.observation_space.shape[0]
        # TODO: play around with state sizes
        self.state_size = 26
        self.action_size = action_space
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, action_space):
        super(Critic, self).__init__()
        #self.state_size = env.observation_space.shape[0]
        self.state_size = 26
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1) # Note the single value - this is V(s)!

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


class ActorCritic(Agent):
    """
    One-step Actor-Critic based off the Assignment 5 code.
    """
    def __init__(self, name, training_mode, action_space, actor_lr, critic_lr, gamma):
        super().__init__(name, training_mode)
        self.actor = Actor(action_space)
        self.critic = Critic(action_space)
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def load_data(self, filepath):
        pass

    def save_data(self, filepath):
        pass

    def start_episode(self):
        self.rewards = []
        self.states = []

    def end_episode(self):
        pass

    def next_move(self, state):
        combined_state = np.concatenate((state[0], state[1]), axis=None)
        self.states.append(combined_state)
        tensor_state = torch.FloatTensor(combined_state).to("cpu")
        dist = self.actor(tensor_state)
        action = dist.sample()
        return action

    def update_agent(self, state, action, reward, done):
        """
        Due to the next_move/update_agent split we use the terminology "previous_state, state"
        instead of "state, next_state".
        """
        if not self.training_mode:
            return

        previous_state = torch.FloatTensor(self.states[-1]).to("cpu")
        previous_value = self.critic(previous_state)
        if not done:
            combined_state = np.concatenate((state[0], state[1]), axis=None)
            state = torch.FloatTensor(combined_state).to("cpu")
            value = self.critic(state)
        else:
            value = 0

        dist = self.actor(previous_state)
        tensor_action = torch.tensor(action, dtype=torch.int32)
        log_prob = dist.log_prob(tensor_action).unsqueeze(0)

        delta = reward + (self.gamma * value) - previous_value
        actor_loss = -((log_prob * delta.detach()).mean())
        critic_loss = delta.pow(2).mean()

        # update w
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # update theta
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

