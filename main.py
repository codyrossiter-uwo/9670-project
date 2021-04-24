import random

import numpy as np

from agents.actor_critic import ActorCritic
from agents.agent import Agent
from agents.monte_carlo import MonteCarlo
from agents.random_agent import RandomAgent
from curling_discrete import CurlingEnv
from game import Game
from player_coordinator import PlayerCoordinator

import matplotlib.pyplot as plt

from agents.td_zero import TDZero

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
    """
    agent = MonteCarlo("Monte Carlo",
                       training_mode=True,
                       action_space=game.get_environment().action_space.n,
                       gamma=0.48542,
                       epsilon=0.94587,
                       decay_rate=0.999)
   """
    agent = ActorCritic("Actor Critic",
                        training_mode=True,
                        action_space=game.get_environment().action_space.n,
                        actor_lr=0.01,
                        critic_lr=0.01,
                        gamma=0.5)
    filepath = "monte.json"
    #agent.load_data(filepath)
    opponent = RandomAgent("Random", False, game.get_environment().action_space.n)

    winner = game.play(agent, opponent, render=True)

    if winner:
        print("Winner is {}".format(winner.name))
    else:
        print("Draw")

    #agent.save_data(filepath)

