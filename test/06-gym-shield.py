import gym
import gym_ergojr
import time
import numpy as np

env = gym.make("ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0")
env.reset()

for i in range(1000):
    if i % 50 == 0:
        action = env.action_space.sample()
    obs, rew, done, _ = env.step(action)
    if rew == 1:
        print(i, np.around(obs, 1), rew)

    time.sleep(0.01)
