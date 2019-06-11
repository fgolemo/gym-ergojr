import time
import gym
from tqdm import trange

import gym_ergojr

env = gym.make("ErgoGripper-Square-Headless-v1")

for i in trange(120):
    env.reset()

env.render('human')
time.sleep(2)
