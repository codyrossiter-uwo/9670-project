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
ROUNDS = 10

ONE_POINT_TILES = [(0, 1), (1, 0), (1, 2), (2, 1)]
THREE_POINT_TILES = [(1, 1)]

class CurlingEnv(gym.Env):
    """
    TODO: High-level env overview

    TODO: May need to add rounds to state: eg r1 [001,010,200]

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
        self.last_action = None

        self.action_space = gym.spaces.Discrete(SHOT_LOCATIONS * POWER_LEVELS)

    def reset(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.player_counter = 1
        self.round_counter = 0
        self.last_action = None

        return self.grid

    def next_player(self):
        self.player_counter += 1
        if self.player_counter > 2:
            self.player_counter = 1

    def step(self, action):
        # Process the shot
        location = int(action / SHOT_LOCATIONS)
        target = int(action % SHOT_LOCATIONS)

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

        # update our last action for our render method
        self.last_action = (action, self.player_counter)

        self.next_player()

        self.round_counter += 1
        done = self.round_counter >= ROUNDS

        if done:
            reward = self.calculate_reward()
        else:
            reward = (0, 0)

        return self.grid, reward, done, {}

    def calculate_reward(self):
        score_1 = 0
        score_2 = 0

        for x, y in ONE_POINT_TILES:
            if self.grid[x][y] == 1:
                score_1 += 1
            elif self.grid[x][y] == 2:
                score_2 += 1

        for x, y in THREE_POINT_TILES:
            if self.grid[x][y] == 1:
                score_1 += 3
            elif self.grid[x][y] == 2:
                score_2 += 3

        #return score_1 - score_2, score_2 - score_1

        # TODO: test with 1 for winning -1 for drawing or losing
        if score_1 > score_2:
            return 1, -1
        elif score_2 > score_1:
            return -1, 1
        else:
            return -1, -1

    def render(self, mode='human'):
        print()
        if self.last_action:
            location = int(self.last_action[0] / 3)
            target = int(self.last_action[0] % 3)
            print("Player {} threw to  {} from {}".format(self.last_action[1], target, location))
        for row in self.grid:
            print(row)
