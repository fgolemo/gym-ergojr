import gym
import numpy as np
import torch
import os
from torch import load

from gym_ergojr.models.model_reacher_v2 import ReacherModelV2


class ErgoReacherPlus2Wrapper(gym.Wrapper):
    def __init__(self, env, nosim=False):
        super().__init__(env)
        self.env = env
        self.nosim = nosim

        HIDDEN_NODES = 100

        if not self.nosim:
            modelFile = "../trained_lstms/lstm_ers_v3_exp6_l3_n{}.pt".format(HIDDEN_NODES)
            modelArch = ReacherModelV2(
                hidden_cells=HIDDEN_NODES
            )
        else:
            modelFile = "../trained_lstms/lstm_ers_ff_v1_exp1_l3_n{}.pt".format(HIDDEN_NODES)
            modelArch = ReacherModelV2(
                hidden_cells=HIDDEN_NODES,
                ff=True
            )

        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), modelFile)
        self.load_model(modelArch, full_path)
        self.step_counter = 0

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location="cpu")
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        print("DBG: MODEL LOADED:", modelPath)

    def data_to_var(self, sim_t2, real_t1, action):
        cat = []
        if not self.nosim:
            cat.append(torch.from_numpy(sim_t2).float())
        cat.append(torch.from_numpy(real_t1).float())
        cat.append(torch.from_numpy(action).float())

        out = torch.cat(cat, dim=0)
        return out

    def get_parameters(self):
        return self.net.parameters()

    def step(self, action):
        obs_real_t1 = self.unwrapped._get_obs()

        obs_sim_t2, _, _, info = self.unwrapped.step(action)

        with torch.no_grad():
            obs_sim_t2_ = None
            if not self.nosim:
                obs_sim_t2_ = obs_sim_t2[:8].copy()

            variable = self.data_to_var(obs_sim_t2_, obs_real_t1[:8].copy(), np.array(action).copy())
            obs_real_t2_delta = self.net.infer(variable)

        if not self.nosim:
            obs_real_t2 = obs_sim_t2[:8].copy() + obs_real_t2_delta
        else:
            obs_real_t2 = obs_real_t1[:8].copy() + obs_real_t2_delta

        self.unwrapped._set_state(obs_real_t2)

        new_obs = np.zeros((10), dtype=np.float32)
        new_obs[:8] = obs_real_t2  # set joint pos/vel
        new_obs[8:] = obs_sim_t2[8:]  # set target x/y pos

        # print("real t1:", obs_real_t1[:8].round(2))
        # print("sim_ t2:", obs_sim_t2[:8].round(2))
        # print("action_:", np.around(action,2))
        # print("real t2:", obs_real_t2[:8].round(2))
        # print("===")

        self.step_counter += 1

        reward, done = self.unwrapped._getReward()
        if self.step_counter >= self.env._max_episode_steps:
            _ = self.reset()
            done = True

        return new_obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        self.net.zero_hidden(1)  # !important
        return self.env.reset()

    def set_state(self, state):
        return self.unwrapped._set_state(state)


def ErgoReacherPlus2Env(base_env_id, nosim=False):
    return ErgoReacherPlus2Wrapper(gym.make(base_env_id), nosim)


if __name__ == '__main__':
    import gym_ergojr
    import time

    env = gym.make("ErgoReacher-Graphical-Simple-Plus-NoSim-v2")

    env.reset()

    # for episode in range(10):
    for step in range(1005):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        # time.sleep(0.01)
        print(step, rew, done, info)
    # env.reset()
