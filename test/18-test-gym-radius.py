import gym
import gym_ergojr
import time
from gym_ergojr.envs import ErgoReacherEnv
import numpy as np

recording = np.zeros((6, 120, 4))  # 4 settings, 30 time steps, 2 joints + 2 vels

env = ErgoReacherEnv(headless=False, simple=True, goal_halfsphere=True)

# env.robot.set_text("hellooooooo")

for i in range(100):
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())
    # recording[setting_idx, i+(action_idx*30), :] = obs[[0, 1, 4, 5]]
    time.sleep(0.1)

env.close()

# import matplotlib.pyplot as plt
#
# for i in range(6):
#     plt.subplot(6, 1, 1 + i)
#     plt.plot(np.arange(120), recording[i, :, 0], label="joint 0")
#     plt.plot(np.arange(120), recording[i, :, 1], label="joint 1")
#     plt.plot(np.arange(120), recording[i, :, 2], label="vel 0")
#     plt.plot(np.arange(120), recording[i, :, 3], label="vel 1")
#     plt.legend()
#
# plt.show()

