import numpy as np
import pickle
import random
from collections import defaultdict

from agents.agent import Agent


def epsilon_greedy_policy(action_space, Q, state, epsilon, optimal_actions):
    # If we know the optimal action, we use the updated probabilities to randomly select it or
    # the sub-optimal actions.
    hashable_state = pickle.dumps(state)
    if optimal_actions[hashable_state]:
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
    def __init__(self, name, gamma=1, epsilon=0.1, decay_rate=0.99):
        super().__init__(name)
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.Q = defaultdict(lambda: np.random.random(9))
        self.optimal_actions = defaultdict(lambda: defaultdict(list))


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
        # TODO: remove hardcoded action space
        action = epsilon_greedy_policy(9, self.Q, state, self.epsilon, self.optimal_actions)
        return action

    def update_agent(self, state, action, reward, done):

        # store our Si, Ai, Ri+1
        self.states.append(state)
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
                    self.optimal_actions[self.states[t]] = a_star

            # add the episode reward sum for our graph
            # cumulative_rewards.append(np.sum(rewards))


