import gym
import numpy as np
import torch
import os
from torch import load

from gym_ergojr.models.model_lstm_v5 import LstmNetRealv5


class ErgoReacherPlusWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        model = "1"

        modelFile = "../trained_lstms/lstm_ergoreachersimple_v{}_exp1_l3_n128.pt".format(model)
        modelArch = LstmNetRealv5(
            n_input_state_sim=8,
            n_input_state_real=8,
            n_input_actions=4,
            nodes=128,
            layers=3,
            cuda=False
        )

        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), modelFile)
        self.load_model(modelArch, full_path)
        self.step_counter = 0

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location="cpu")
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
        torch.no_grad()
        print("DBG: MODEL LOADED:", modelPath)

    @staticmethod
    def double_unsqueeze(data):
        return torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)

    @staticmethod
    def double_squeeze(data):
        return torch.squeeze(torch.squeeze(data)).data.cpu().numpy()

    def data_to_var(self, sim_t2, real_t1, action):
        out = self.double_unsqueeze(torch.cat(
            [torch.from_numpy(sim_t2).float(),
             torch.from_numpy(real_t1).float(),
             torch.from_numpy(action).float()], dim=0))
        return out

    def data_to_var_nosim(self, real_t1, action):
        out = self.double_unsqueeze(torch.cat(
            [torch.from_numpy(real_t1).float(),
             torch.from_numpy(action).float()], dim=0))
        return out

    def get_parameters(self):
        return self.net.parameters()

    def step(self, action):
        self.step_counter += 1
        obs_real_t1 = self.unwrapped._get_obs()

        obs_sim_t2, _, _, info = self.unwrapped.step(action)
        variable = self.data_to_var(obs_sim_t2[:8].copy(), obs_real_t1[:8].copy(), np.array(action).copy())

        obs_real_t2_delta = self.double_squeeze(self.net.forward(variable))

        obs_real_t2 = obs_sim_t2[:8].copy() + obs_real_t2_delta

        new_state = np.zeros((12), dtype=np.float32)
        new_state[[1, 2, 4, 5, 7, 8, 10, 11]] = obs_real_t2
        self.unwrapped._set_state(new_state)

        new_obs = np.zeros((10), dtype=np.float32)
        new_obs[:8] = obs_real_t2
        new_obs[8:] = obs_sim_t2[8:]

        # print("real t1:", obs_real_t1[:8].round(2))
        # print("sim_ t2:", obs_sim_t2[:8].round(2))
        # print("action_:", np.around(action,2))
        # print("real t2:", obs_real_t2[:8].round(2))
        # print("===")

        # done = False

        self.step_counter += 1

        reward, done = self.unwrapped._getReward()
        if self.step_counter >= self.env._max_episode_steps:
            _ = self.reset()
            done = True

        return new_obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        self.net.zero_hidden()  # !important
        self.net.hidden = (self.net.hidden[0].detach(),
                           self.net.hidden[1].detach())
        return self.env.reset()

    # def set_state(self, state):
    #     return self.unwrapped._set_state(state)


def ErgoReacherPlusEnv(base_env_id):
    return ErgoReacherPlusWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_ergojr
    import time

    env = gym.make("ErgoReacher-Graphical-Simple-Plus-v1")

    env.reset()

    # for episode in range(10):
    for step in range(1005):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        # time.sleep(0.01)
        print(step, rew, done, info)
    # env.reset()
