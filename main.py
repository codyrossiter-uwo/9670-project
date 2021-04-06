import random

import numpy as np

from agents.agent import Agent
from agents.random_agent import RandomAgent
from curling_discrete import CurlingEnv
from player_coordinator import PlayerCoordinator

import matplotlib.pyplot as plt

from agents.td_zero import TDZero




if __name__ == '__main__':
    env = CurlingEnv()
    """
    agent1 = MonteCarlo("Monte",
                        gamma=0.7185,
                        epsilon=0.96599,
                        decay_rate=0.98786)
    """
    agent1 = TDZero(
        "Zero",
        alpha=0.93917,
        epsilon=0.94059,
        gamma=0.11040,
        decay_rate=0.96481)
    agent2 = RandomAgent("Random", 9)

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

