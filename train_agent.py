"""
This script creates or loads an agent trains it against another agent.
"""
from agents.actor_critic import ActorCritic
from agents.monte_carlo import MonteCarlo
from agents.random_agent import RandomAgent
from agents.td_zero import TDZero
from game import Game
from tqdm import tqdm
import numpy as np

NUM_GAMES = 10000

if __name__ == '__main__':
    game = Game(hard_mode=True)
    td_zero = TDZero("Zero",
                   training_mode=True,
                   alpha=0.4179,
                   gamma=0.48542,
                   epsilon=0.94587,
                   decay_rate=0.9658)
    td_zero.load_data("td_zero.json")

    monte_carlo = MonteCarlo("Monte",
                   training_mode=True,
                   action_space=game.get_environment().action_space.n,
                   gamma=0.7364,
                   epsilon=0.68118,
                   decay_rate=0.56880)
    monte_carlo.load_data("monte.json")

    ac = ActorCritic("Actor Critic",
                        training_mode=True,
                        action_space=game.get_environment().action_space.n,
                        actor_lr=0.06215,
                        critic_lr=0.103795,
                        gamma=0.3501)

    random_agent = RandomAgent("Random", False, game.get_environment().action_space.n)

    agent = td_zero
    opponent = random_agent

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

    # save the data only for our training agent
    if agent is monte_carlo:
        monte_carlo.save_data("monte.json")
    elif agent is td_zero:
        td_zero.save_data("td_zero.json")

    last_quarter_win_percentage = np.mean(wins[-int(NUM_GAMES/4):])

    print("Total wins", np.sum(wins))
    print("Mean value for wins", np.mean(wins))
    print("Win mean value for the last quarter", last_quarter_win_percentage)

