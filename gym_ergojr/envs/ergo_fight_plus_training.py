import gym
import numpy as np
import torch
import os
from gym_ergojr.models.model_lstm_v3 import LstmNetRealv3
from torch import load
from torch.autograd import Variable


class ErgoFightPlusTrainingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.step_counter = 0

    def set_net(self, nodes, layers):
        self.net = LstmNetRealv3(nodes=nodes, layers=layers, cuda=True)
        self.net = self.net.cuda()

    @staticmethod
    def double_unsqueeze(data):
        return torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)

    @staticmethod
    def double_squeeze(data):
        return torch.squeeze(torch.squeeze(data)).data.cpu().numpy()

    def _to_input(self, sim_t2, real_t1, action):
        return Variable(
            self.double_unsqueeze(torch.cat(
                [torch.from_numpy(sim_t2).float(),
                 torch.from_numpy(real_t1).float(),
                 torch.from_numpy(action).float()], dim=0))).cuda()

    def _to_var(self, data):
        return Variable(torch.from_numpy(data)).cuda()

    def get_parameters(self):
        return self.net.parameters()

    def step(self, action):
        self.step_counter += 1
        obs_real_t1 = self.unwrapped._self_observe()

        obs_sim_t2, _, _, info = self.unwrapped.step(action, dry_run=True)
        variable = self._to_input(obs_sim_t2[:12].copy(), obs_real_t1[:12].copy(), np.array(action).copy())

        obs_real_t2_delta = self.net.forward(variable)
        obs_real_t2 = self._to_var(obs_sim_t2[:12].copy()).float() + obs_real_t2_delta

        new_obs = self.unwrapped.set_state(obs_real_t2.data.cpu().numpy().flatten())

        # print("real t1:", obs_real_t1[:12].round(2))
        # print("sim_ t2:", obs_sim_t2[:12].round(2))
        # print("action_:", np.around(action,2))
        # print("real t2:", obs_real_t2[:12].round(2))
        # print("===")

        done = False
        if self.step_counter >= self.env._max_episode_steps:
            _ = self.reset()  # automatically reset the env
            done = True  # this is nasty but IDK how else to do it

        return new_obs, self.unwrapped._getReward(), done, {"new_state": obs_real_t2}

    def reset(self):
        self.step_counter = 0
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        self.net.zero_grad()
        return self.env.reset()

    def set_state(self, state):
        self.unwrapped.set_state(state)


def ErgoFightPlusTrainingEnv(base_env_id):
    return ErgoFightPlusTrainingWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_ergojr
    import time

    # env = gym.make("ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-Plus-v0")
    #
    # env.reset()
    #
    # # for episode in range(10):
    # for step in range(1005):
    #     action = env.action_space.sample()
    #     obs, rew, done, info = env.step(action)
    #     # time.sleep(0.01)
    #     print(step, rew, done, info)
    # # env.reset()
