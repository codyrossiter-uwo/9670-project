import sys
from contextlib import closing
import numpy as np
import gym
from gym import utils
from gym.envs.toy_text import discrete
from io import StringIO

GRID_SIZE = 3
SHOT_LOCATIONS = 3
POWER_LEVELS = 3


class CurlingEnv(gym.Env):
    """
    TODO: High-level env overview

    Here the state is represented by a 3x3 grid where each space can have
    a 0 (no rock), 1 (player 1's rock), or 2 (player 2's rock) like so:

    0 0 1
    2 2 0
    1 1 0

    TODO: Determine number of rounds
    Rewards are given at the end of the episode based on the points scored.
    The center square gives a reward of 3, the positions to the left/right/above/below
    give a reward of 1, and no rewards are given for corner positions.
    The reward grid looks like:

    0 1 0
    1 3 1
    0 1 0
    """

    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.player_counter = 1

    def next_player(self):
        self.player_counter += 1
        if self.player_counter > 2:
            self.player_counter = 1

    def step(self, action):
        # Process the shot
        # TODO: add power level. We'll start with just location for now and randomly
        #       place the shot
        location = action
        random_number = np.random.random()
        if random_number < 0.25:
            target = 0
        elif 0.25 <= random_number < 0.75:
            target = 1
        else:
            target = 2

        # check to see if the shot is clear
        # TODO: maybe put the check and push back code in the same for loop
        blocked_tile = None
        for i in range(target + 1):
            if self.grid[location][target] != 0:
                blocked_tile = i
                break

        # Place the rock directly if their is no blocker
        if blocked_tile is None:
            self.grid[location][target] = self.player_counter
        # Otherwise push all rocks back 1 tile
        else:
            # consider location 1 is blocked
            incoming_rock = self.player_counter
            while blocked_tile





        self.next_player()
        return None


    def render(self, mode='human'):
        pass
