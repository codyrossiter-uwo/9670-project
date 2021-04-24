# 9670-project

This project contains code for a curling environment where two agents
compete to get the highest score.

The agents take turns throwing rocks and after a certain number of turns
the scores are calculated and the winner is chosen.

This environment is considered solved when a trained agent can defeat the random agent with a win rate of at least 60%.

## Main Scripts

There are three main scripts for training and running the agents:
* `main.py` which contains the script to run one game and display the turn-by-turn action.
* `train_agent` which runs one agent against another for several thousand games in order to train it.
* `optimize.py` which contains an optuna optimization script for tuning agent hyperparameters.

## Agents

There are four agents to choose from:
* Random Agent: This agent selects random actions and is used mostly for training.
* Monte Carlo: This agent uses the Monte Carlo RL algorithm.
* TD Zero: This agent uses TD zero as its RL algorithm.
* Actor-Critic: This agent uses Actor-Critic one step as it's RL algorithm and uses a neural network for its function approximations.

## Installation

No further installation should be needed after installing the pip libraries
in requirements.txt with `pip install -r requirements.txt`.
Note that this code was written with python 3.9 as the target version.
