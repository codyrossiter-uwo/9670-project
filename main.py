import random
from curling_discrete import CurlingEnv
from player_coordinator import PlayerCoordinator


class RandomAgent:
    def __init__(self):
        self.name = str(random.randint(0, 100))

    def next_move(self):
        return random.randint(0, 2)


if __name__ == '__main__':
    env = CurlingEnv()
    state = env.reset()

    coordinator = PlayerCoordinator(RandomAgent(), RandomAgent(), state)

    done = False
    env.render()
    while not done:
        action = coordinator.next_move()
        state, reward, done, _ = env.step(action)
        coordinator.inform_agent(state, reward, done)
        coordinator.next_turn()
        env.render()

        if done:
            print("Player 1 score: {}".format(reward[0]))
            print("Player 2 score: {}".format(reward[1]))



