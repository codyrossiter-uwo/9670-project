"""
This script creates or loads an agent trains it against another agent.
"""
from agents.monte_carlo import MonteCarlo
from agents.random_agent import RandomAgent
from agents.td_zero import TDZero
from game import Game
from tqdm import tqdm
import numpy as np

NUM_GAMES = 100000

if __name__ == '__main__':
    mean_wins = []
    game = Game()
    """
    agent = TDZero("Zero",
                   training_mode=True,
                   alpha=0.4179,
                   gamma=0.48542,
                   epsilon=0.94587,
                   decay_rate=0.9658)
    """
    agent = MonteCarlo("Zero",
                   training_mode=True,
                   action_space=game.get_environment().action_space.n,
                   gamma=0.48542,
                   epsilon=0.94587,
                   decay_rate=0.999)
    filepath = "monte.json"
    agent.load_data(filepath)
    opponent = RandomAgent("Random", False, game.get_environment().action_space.n)

    wins = []
    for _ in tqdm(range(NUM_GAMES)):
        # shuffle who plays first
        if np.random.random() < 0.5:
            winner = game.play(agent, opponent)
        else:
            winner = game.play(opponent, agent)

        if winner == agent:
            wins.append(1)
        else:
            wins.append(0)

    agent.save_data(filepath)
    mean_wins.append(np.mean(wins))

    print(mean_wins)

