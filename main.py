import random

import numpy as np

from agents.agent import Agent
from agents.monte_carlo import MonteCarlo
from agents.random_agent import RandomAgent
from curling_discrete import CurlingEnv
from game import Game
from player_coordinator import PlayerCoordinator

import matplotlib.pyplot as plt

from agents.td_zero import TDZero


"""
TODO: write a script for how I would train one agent against another for X episodes
      Then, this can be used to write the game manager methods.
"""


if __name__ == '__main__':
    game = Game(hard_mode=True)
    """
    agent = TDZero("Zero",
                   training_mode=True,
                   alpha=0.4179,
                   gamma=0.48542,
                   epsilon=0.94587,
                   decay_rate=0.9658)
    """
    agent = MonteCarlo("Monte Carlo",
                       training_mode=True,
                       action_space=game.get_environment().action_space.n,
                       gamma=0.48542,
                       epsilon=0.94587,
                       decay_rate=0.999)
    filepath = "monte.json"
    #agent.load_data(filepath)
    opponent = RandomAgent("Random", False, game.get_environment().action_space.n)

    # TODO: add back in shuffle code and format output to show agent name
    winner = game.play(agent, opponent, render=True)

    if winner:
        print("Winner is {}".format(winner.name))
    else:
        print("Draw")

    #agent.save_data(filepath)

    """
    print("Wins {}".format(np.sum(wins)))
    plt.plot(rolling_average)
    plt.ylabel("Average over 100")
    plt.xlabel("episode")
    plt.savefig("debug-result")
    """

