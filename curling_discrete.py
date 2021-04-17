import numpy as np
import gym

EASY_REWARD_GRID = [
    [0, 1, 0],
    [1, 3, 1],
    [0, 1, 0]
]

class CurlingEnv(gym.Env):
    """
    TODO: High-level env overview

    TODO: May need to add rounds to state: eg r1 [001,010,200]

    Here the state is represented by a 4x4 grid where each space can have
    a 0 (no rock), 1 (player 1's rock), or 2 (player 2's rock) like so:

    round 2
    0 0 1 0
    2 2 0 0
    1 1 0 0
    2 0 1 1

    TODO: Determine number of rounds
    Rewards are given at the end of the episode based on the points scored.
    The center square gives a reward of 3, the positions to the left/right/above/below
    give a reward of 1, and no rewards are given for corner positions.
    The reward grid looks like:

    1 0 0 1
    0 3 2 0
    0 2 3 0
    1 0 0 1
    """

    def __init__(self, hard_mode=False):
        self.grid_size = 3 if not hard_mode else 5
        self.shot_locations = 3 if not hard_mode else 5
        self.power_levels = 3 if not hard_mode else 5
        self.rounds_to_play = 3 if not hard_mode else 5
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # TODO make hard reward grid
        self.reward_grid = EASY_REWARD_GRID if not hard_mode else None
        self.player_counter = 1
        self.turn_counter = 0
        self.last_action = None

        self.action_space = gym.spaces.Discrete(self.shot_locations * self.power_levels)

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.player_counter = 1
        self.turn_counter = 0
        self.last_action = None

        return [self.turn_counter, self.grid]

    def next_player(self):
        self.player_counter += 1
        if self.player_counter > 2:
            self.player_counter = 1

    def step(self, action):
        # Process the shot
        location = int(action / self.shot_locations)
        target = int(action % self.power_levels)

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
            for i in range(blocked_tile, self.grid_size):
                # if the tile isn't empty swap the rocks
                if rock_to_place != 0:
                    rock_to_place, self.grid[location][i] = self.grid[location][i], rock_to_place

        # update our last action for our render method
        self.last_action = (action, self.player_counter)

        self.next_player()

        self.turn_counter += 1
        # each round has two turns
        done = self.turn_counter >= (self.rounds_to_play * 2)

        if done:
            reward = self.calculate_reward()
        else:
            reward = (0, 0)

        return [self.turn_counter, self.grid], reward, done, {}

    def calculate_reward(self):
        score_1 = 0
        score_2 = 0

        for x, row in enumerate(self.reward_grid):
            for y, value in enumerate(row):
                if self.grid[x][y] == 1:
                    score_1 += value
                elif self.grid[x][y] == 2:
                    score_2 += value

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
            location = int(self.last_action[0] / self.shot_locations)
            target = int(self.last_action[0] % self.power_levels)
            print("Player {} threw to  {} from {}".format(self.last_action[1], target, location))
        for row in self.grid:
            print(row)
