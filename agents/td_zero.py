import json
import pickle
from collections import defaultdict
from json import JSONDecodeError

from agents.agent import Agent
import numpy as np

from helper import state_to_string


def epsilon_greedy_policy(action_space, Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, action_space - 1)
    else:
        hashable_state = state_to_string(state)
        return np.argmax(Q[hashable_state])


class TDZero(Agent):
    def __init__(self, name, training_mode, alpha=0.5, epsilon=0.9, gamma=0.9, decay_rate=0.9):
        super().__init__(name, training_mode)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

        # convert to list to make json saving/loading easier
        self.Q = defaultdict(lambda: list(np.random.random(16)))
        self.V = defaultdict(lambda: 0)
        self.previous_state = None

    def load_data(self, filepath):
        with open(filepath, "r") as file:
            try:
                data = json.load(file)
                self.Q.update(data["Q"])
                self.V.update(data["V"])
            except JSONDecodeError:
                print("JSON data for {} not found. Initializing with empty values".format(self.name))


    def save_data(self, filepath):
        # build the json entry
        data = {
            "Q": dict(self.Q),
            "V": dict(self.V)
        }
        with open(filepath, "w") as file:
            json.dump(data, file)


    def next_move(self, state):
        # The action taken here will move the agent to the next state
        # store this one so we can process it later.
        self.previous_state = state
        # TODO: remove hard-coded state space
        return epsilon_greedy_policy(16, self.Q, state, self.epsilon)

    def start_episode(self):
        self.previous_state = None

    def end_episode(self):
        self.epsilon = min(0.001, self.epsilon * self.decay_rate)

    def update_agent(self, state, action, reward, done):
        if not self.training_mode:
            return

        # Make a hashable version of the states
        s = state_to_string(self.previous_state)
        s_prime = state_to_string(state)

        self.V[s] += self.alpha * (reward + (self.gamma * self.V[s_prime]) - self.V[s])
        self.previous_state = state