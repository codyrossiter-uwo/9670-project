import json
from json import JSONDecodeError

import numpy as np
import pickle
import random
from collections import defaultdict

from agents.agent import Agent
from helper import state_to_string


def epsilon_greedy_policy(action_space, Q, state, epsilon, optimal_actions):
    # If we know the optimal action, we use the updated probabilities to randomly select it or
    # the sub-optimal actions.
    hashable_state = state_to_string(state)
    if hashable_state in optimal_actions:
        if np.random.random() < 1 - epsilon + (epsilon / action_space):
            return optimal_actions[hashable_state]
        else:
            # randomly select an action that isn't the optimal action
            possible_actions = [*range(action_space)]
            possible_actions.remove(optimal_actions[hashable_state])
            return np.random.choice(possible_actions)
    # Otherwise we use regular epsilon-greedy to select a random action
    else:
        if np.random.random() < epsilon:
            return random.randint(0, action_space - 1)
        else:
            return np.argmax(Q[hashable_state])


class MonteCarlo(Agent):
    def __init__(self, name, training_mode, action_space, gamma=1, epsilon=0.1, decay_rate=0.99):
        super().__init__(name, training_mode)
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.Q = defaultdict(lambda: list(np.random.random(action_space)))
        self.optimal_actions = defaultdict(None)

    def load_data(self, filepath):
        with open(filepath, "r") as file:
            try:
                data = json.load(file)
                self.Q.update(data["Q"])
                self.optimal_actions.update(data["optimal_actions"])
            except JSONDecodeError:
                print("JSON data for {} not found. Initializing with empty values".format(self.name))

    def save_data(self, filepath):
        # build the json entry
        data = {
            "Q": dict(self.Q),
            "optimal_actions": dict(self.optimal_actions)
        }
        with open(filepath, "w") as file:
            json.dump(data, file)


    def start_episode(self):
        # set up the arrays to hold the states, actions, and rewards for each time step
        self.states = []
        self.actions = []
        # add in the reward for R0, this won't be used but will help make future array
        # references consistent.
        self.rewards = [0]
        self.returns = defaultdict(lambda: defaultdict(list))

    def end_episode(self):
        # TODO: make this reusable
        self.epsilon *= self.decay_rate

    def next_move(self, state):
        action = epsilon_greedy_policy(self.action_space, self.Q, state, self.epsilon, self.optimal_actions)
        return action

    def update_agent(self, state, action, reward, done):
        if not self.training_mode:
            return

        # store our Si, Ai, Ri+1
        self.states.append(state_to_string(state))
        self.actions.append(action)
        self.rewards.append(reward)

        if done:
            # process the episode
            G = 0
            # The states array has T-1 entries
            for t in range(len(self.states) - 1, 0, -1):
                G = (self.gamma * G) + self.rewards[t + 1]
                # unless the St, At  pair appears in the preceding states and actions
                if self.states[t] not in self.states[:t] and self.actions[t] not in self.actions[:t]:
                    self.returns[self.states[t]][self.actions[t]].append(G)
                    self.Q[self.states[t]][self.actions[t]] = np.average(self.returns[self.states[t]][self.actions[t]])
                    a_star = np.argmax(self.Q[self.states[t]])
                    # add the optimal action, the probabilities are handled in the policy function
                    self.optimal_actions[self.states[t]] = int(a_star)

            # add the episode reward sum for our graph
            # cumulative_rewards.append(np.sum(rewards))


