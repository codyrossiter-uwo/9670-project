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
ROUNDS = 4


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
        self.round_counter = 0

        # TODO: convert action to (location, power) pair
        # self.action_space = gym.spaces.MultiDiscrete([SHOT_LOCATIONS, POWER_LEVELS])
        self.action_space = gym.spaces.Discrete(SHOT_LOCATIONS)

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
        blocked_tile = None
        for i in range(target + 1):
            if self.grid[location][i] != 0:
                blocked_tile = i
                break

        # Place the rock directly if their is no blocker
        if blocked_tile is None:
            self.grid[location][target] = self.player_counter
        # Otherwise push all rocks back 1 tile
        else:
            rock_to_place = self.player_counter
            for i in range(blocked_tile, GRID_SIZE):
                # if the tile isn't empty swap the rocks
                if rock_to_place != 0:
                    rock_to_place, self.grid[location][i] = self.grid[location][i], rock_to_place

        self.next_player()

        self.round_counter += 1
        done = self.round_counter >= ROUNDS

        if done:
            # TODO: calculate reward based on score
            reward = 10
        else:
            reward = 0

        # TODO: right now we assume only player 1 can train (eg, receive rewards)
        #       maybe rewrite so that each agent is told what player they are
        #       so they can be like player 0 gets reward[0], player 1 gets reward[1]
        return self.grid, reward, done, {}

    def render(self, mode='human'):
        print()
        for row in self.grid:
            print(row)
