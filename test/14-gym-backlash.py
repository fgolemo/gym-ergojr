import gym
import gym_ergojr
import time
import numpy as np

env = gym.make("ErgoReacher-Graphical-Simple-Backlash-v1")
env.reset()
env.seed(15)

for i in range(1005):
    if i % 50 == 0:
        action = env.action_space.sample()
    obs, rew, done, _ = env.step(action)
    # if rew == 1:
    #     print(i, np.around(obs, 1), rew)

    print(i, rew, done)

    time.sleep(0.01)
