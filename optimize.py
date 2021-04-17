import random

import numpy as np
import optuna

from agents.monte_carlo import MonteCarlo
from agents.random_agent import RandomAgent
from curling_discrete import CurlingEnv
from player_coordinator import PlayerCoordinator
from agents.td_zero import TDZero

"""
This script uses the Optuna library to optimize the hyperparameters as shown
in the colab notebook: 
https://colab.research.google.com/drive/1uRCh8SvpVars-oxyL1t4dBxbXm70F29v?usp=sharing#scrollTo=faK-vvkceHod
"""


def objective(trial):
    env = CurlingEnv(hard_mode=True)
    """
    agent1 = TDZero(str(random.randint(0, 100)),
                        training_mode=True,
                        alpha=trial.suggest_float('alpha', 0.1, 1.0),
                        gamma=trial.suggest_float('gamma', 0.1, 1.0),
                        epsilon=trial.suggest_float('epsilon', 0.9, 1.0),
                        decay_rate=trial.suggest_float('decay_rate', 0.9, 0.99999))
    """
    agent1 = MonteCarlo("Monte Carlo",
                        training_mode=True,
                        action_space=env.action_space.n,
                        gamma=trial.suggest_float('gamma', 0.1, 1.0),
                        epsilon=trial.suggest_float('epsilon', 0.1, 1.0),
                        decay_rate=trial.suggest_float('decay_rate', 0.5, 0.9999))
    agent2 = RandomAgent("Random", False, env.action_space.n)

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
                if reward[0] > reward[1]:
                    wins.append(1)
                else:
                    wins.append(0)

        coordinator.end_episode()
        if len(wins) > 100:
            rolling_average.append(np.mean(wins[-100:]))


    score = np.mean(rolling_average[-5000:])
    return score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

print(study.best_trial)