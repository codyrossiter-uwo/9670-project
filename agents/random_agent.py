import random
from agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, name, action_space):
        super().__init__(name)
        self.action_space = action_space

    def next_move(self, state):
        return random.randint(0, self.action_space - 1)

    def update_agent(self, state, action, reward, done):
        # No need to update the state or reward.
        pass

    def start_episode(self):
        # No state to initialize
        pass

    def end_episode(self):
        # No state or variable updates needed.
        pass
