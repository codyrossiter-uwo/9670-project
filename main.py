import random

from agent import Agent
from curling_discrete import CurlingEnv
from player_coordinator import PlayerCoordinator


class RandomAgent(Agent):

    def next_move(self):
        return random.randint(0, 2)

    def update_agent(self, state, reward, done):
        pass


if __name__ == '__main__':
    env = CurlingEnv()
    agent1 = RandomAgent(str(random.randint(0, 100)))
    agent2 = RandomAgent(str(random.randint(0, 100)))

    for _ in range(10):
        state = env.reset()
        coordinator = PlayerCoordinator(agent1, agent2, state)
        done = False
        while not done:
            action = coordinator.next_move()
            state, reward, done, _ = env.step(action)
            coordinator.inform_player(state, reward, done)
            coordinator.next_turn()



