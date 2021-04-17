from curling_discrete import CurlingEnv
from player_coordinator import PlayerCoordinator

class Game:
    def __init__(self, hard_mode=False):
        self.env = CurlingEnv()
        self.hard_mode = hard_mode

    def get_environment(self):
        return self.env

    def play(self, agent1, agent2, render=False):
        env = CurlingEnv(self.hard_mode)
        state = env.reset()
        coordinator = PlayerCoordinator(agent1, agent2, state)
        coordinator.start_episode()
        done = False
        winner = None
        if render:
            env.render()
        while not done:
            action = coordinator.next_move(state)
            state, reward, done, _ = env.step(action)
            coordinator.inform_players(state, action, reward, done)
            coordinator.next_turn()

            if render:
                env.render()

            if done:
                if reward[0] > reward[1]:
                    winner = agent1
                elif reward[1] > reward[0]:
                    winner = agent2
                else:
                    winner = None

        coordinator.end_episode()

        return winner
