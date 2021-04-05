from collections import defaultdict

from agent import Agent
import pickle
import numpy as np


def epsilon_greedy_policy(action_space, Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, action_space - 1)
    else:
        hashable_state = pickle.dumps(state)
        return np.argmax(Q[hashable_state])


class TDZero(Agent):
    def __init__(self, name, alpha=0.5, epsilon=0.9, gamma=0.9, decay_rate=0.9):
        super().__init__(name)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

        self.Q = defaultdict(lambda: np.random.random(9))
        self.V = defaultdict(lambda: 0)
        self.previous_state = None

    def next_move(self, state):
        # The action taken here will move the agent to the next state
        # store this one so we can process it later.
        self.previous_state = state
        # TODO: remove hard-coded state space
        return epsilon_greedy_policy(9, self.Q, state, self.epsilon)

    def start_episode(self):
        self.previous_state = None

    def end_episode(self):
        self.epsilon = min(0.001, self.epsilon * self.decay_rate)

    def update_agent(self, state, action, reward, done):
        # Make a hashable version of the states
        s = pickle.dumps(self.previous_state)
        s_prime = pickle.dumps(state)

        self.V[s] += self.alpha * (reward + (self.gamma * self.V[s_prime]) - self.V[s])
        self.previous_state = state