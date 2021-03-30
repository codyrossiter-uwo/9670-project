import gym
from curling_discrete import CurlingEnv

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = CurlingEnv()

    done = False
    env.render()
    while not done:
        state, reward, done, _ = env.step(env.action_space.sample())
        env.render()



