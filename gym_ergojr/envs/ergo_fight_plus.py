import gym
import numpy as np
import torch
import os
from gym_ergojr.models.model_lstm_v3 import LstmNetRealv3
from torch import load
from torch.autograd import Variable


class ErgoFightPlusWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ErgoFightPlusWrapper, self).__init__(env)
        self.env = env

        model = "../trained_lstms/lstm_real_v3_exp7_l3_n256.pt"
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model)
        self.load_model(LstmNetRealv3(nodes=256, layers=3, cuda=False), full_path)
        self.step_counter = 0

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location="cpu")
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
        print("DBG: MODEL LOADED:", modelPath)

    @staticmethod
    def double_unsqueeze(data):
        return torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)

    @staticmethod
    def double_squeeze(data):
        return torch.squeeze(torch.squeeze(data)).data.cpu().numpy()

    def data_to_var(self, sim_t2, real_t1, action):
        return Variable(
            self.double_unsqueeze(torch.cat(
                [torch.from_numpy(sim_t2).float(),
                 torch.from_numpy(real_t1).float(),
                 torch.from_numpy(action).float()], dim=0)), volatile=True)

    def step(self, action):
        self.step_counter += 1
        obs_real_t1 = self.unwrapped._self_observe()
        obs_sim_t2, _, _, info = self.unwrapped.step(action, dry_run=True)

        variable = self.data_to_var(obs_sim_t2[:12].copy(), obs_real_t1[:12].copy(), np.array(action).copy())

        obs_real_t2_delta = self.double_squeeze(self.net.forward(variable))

        obs_real_t2 = obs_sim_t2[:12].copy() + obs_real_t2_delta

        new_obs = self.unwrapped.set_state(obs_real_t2)

        # print("real t1:", obs_real_t1[:12].round(2))
        # print("sim_ t2:", obs_sim_t2[:12].round(2))
        # print("action_:", action.round(2))
        # print("real t2:", obs_real_t2[:12].round(2))
        # print("===")

        done = False
        if self.step_counter >= self.env._max_episode_steps:
            _ = self.reset()  # automatically reset the env
            done = True # this is nasty but IDK how else to do it

        return new_obs, self.unwrapped._getReward(), done, info

    def reset(self):
        self.step_counter = 0
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        self.net.zero_grad()
        return self.env.reset()


def ErgoFightPlusEnv(base_env_id):
    return ErgoFightPlusWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_ergojr
    import time

    env = gym.make("ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-Plus-v0")

    env.reset()

    # for episode in range(10):
    for step in range(1005):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        # time.sleep(0.01)
        print (step, rew, done, info)
    # env.reset()
