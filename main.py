import random

import numpy as np

from agent import Agent
from curling_discrete import CurlingEnv
from monte_carlo import MonteCarlo
from player_coordinator import PlayerCoordinator

import matplotlib.pyplot as plt

class RandomAgent(Agent):

    def next_move(self, state):
        return random.randint(0, 8)

    def update_agent(self, state, action, reward, done):
        # No need to update the state or reward.
        pass

    def start_episode(self):
        # No state to initialize
        pass

    def end_episode(self):
        # No state or variable updates needed.
        pass


if __name__ == '__main__':
    env = CurlingEnv()
    agent1 = MonteCarlo("Monte",
                        gamma=0.7185,
                        epsilon=0.96599,
                        decay_rate=0.98786)
    agent2 = RandomAgent("Random")

    wins = []
    rolling_average = []

    for _ in range(10000):
        state = env.reset()
        coordinator = PlayerCoordinator(agent1, agent2, state)
        coordinator.start_episode()
        done = False
        while not done:
            action = coordinator.next_move(state)
            state, reward, done, _ = env.step(action)
            coordinator.inform_players(state, action, reward, done)
            coordinator.next_turn()

            if done:
                # print("player 1:", reward[0], "\nplayer 2:", reward[1])
                if reward[0] > reward[1]:
                    wins.append(1)
                else:
                    wins.append(0)

        coordinator.end_episode()
        if len(wins) > 100:
            rolling_average.append(np.mean(wins[-100:]))


    print("Wins {}".format(np.sum(wins)))
    plt.plot(rolling_average)
    plt.ylabel("Average over 100")
    plt.xlabel("episode")
    plt.savefig("debug-result")

