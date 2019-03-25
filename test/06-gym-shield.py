import gym
import gym_ergojr
import time
import numpy as np

env = gym.make("ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0")
env.reset()

start = time.time()

for i in range(10000):
    if i % 50 == 0:
        action = env.action_space.sample()
    obs, rew, done, _ = env.step(action)
    if done:
        env.reset()
    # if rew == 1:
    #     print(i, np.around(obs, 1), rew)

    # print(i, rew, done)

    # time.sleep(0.01)

end = time.time() - start
print ("total time: ",end,". FPS: ",10000/end)