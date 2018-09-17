import numpy as np
import gym
import gym_ergojr
import matplotlib.pyplot as plt

np.random.seed(0)

env_sim1 = gym.make("ErgoReacher-Headless-Simple-v1")
env_sim1.reset()
env_sim2 = gym.make("ErgoReacher-Headless-Simple-v1")
env_sim2.reset()

env_sim1.unwrapped._set_state([0] * 8)
env_sim2.unwrapped._set_state([0] * 8)

traj1 = np.zeros((100, 4), dtype=np.float32)
traj2 = np.zeros((100, 4), dtype=np.float32)
acts1 = np.zeros((100, 4), dtype=np.float32)
acts2 = np.zeros((100, 4), dtype=np.float32)

for i in range(100):
    if i % 10 == 0:
        action1 = np.random.uniform(low=-1, high=1, size=4)
        action2 = np.random.uniform(low=-1, high=1, size=4)

    obs1, _, _, _ = env_sim1.step(action1)
    obs2, _, _, _ = env_sim2.step(action2)

    traj1[i] = obs1[:4]
    traj2[i] = obs2[:4]

    acts1[i] = action1
    acts2[i] = action2

x = np.arange(100)

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')


def add_to_diag(ax):
    ax.plot(x, acts1[:, 0], label="act 1")
    ax.plot(x, traj1[:, 0], label="trj 1")
    ax.plot(x, acts2[:, 0], label="act 2")
    ax.plot(x, traj2[:, 0], label="trj 2")


for i in range(2):
    for j in range(2):
        add_to_diag(ax[i, j])

plt.legend()
plt.show()
