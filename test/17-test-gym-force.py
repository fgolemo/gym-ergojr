import gym
import gym_ergojr
import time
from gym_ergojr.envs import ErgoReacherEnv
import numpy as np

recording = np.zeros((6, 120, 4))  # 4 settings, 30 time steps, 2 joints + 2 vels

for setting_idx, (max_force, max_vel) in enumerate([
    (1, 18),
    (100, 18),
    (1, 30),
    (100, 30),
    (1, 4),
    (.1, 4)
]):
    env = ErgoReacherEnv(headless=False, simple=True, max_force=max_force, max_vel=max_vel)
    env.reset()

    env.robot.set_text("hellooooooo")

    for action_idx, action in enumerate([
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0]
    ]):
        for i in range(30):
            obs, _, _, _ = env.step(action)
            recording[setting_idx, i+(action_idx*30), :] = obs[[0, 1, 4, 5]]
            time.sleep(0.1)

    env.close()

import matplotlib.pyplot as plt

for i in range(6):
    plt.subplot(6, 1, 1 + i)
    plt.plot(np.arange(120), recording[i, :, 0], label="joint 0")
    plt.plot(np.arange(120), recording[i, :, 1], label="joint 1")
    plt.plot(np.arange(120), recording[i, :, 2], label="vel 0")
    plt.plot(np.arange(120), recording[i, :, 3], label="vel 1")
    plt.legend()

plt.show()