import gym
import numpy as np
import torch
from gym_ergojr.envs.ergo_reacher_env import ErgoReacherEnv


class ErgoReacherAugmented(ErgoReacherEnv):
    def __init__(self, model, model_path):
        super(ErgoReacherAugmented, self).__init__()
        self.model = model
        self.model_path = model_path
        self.modified_obs = torch.zeros(1, 12).float()
        self.modified_actions = torch.zeros(1, 6).float()

    def load_model(self):
        return self.model.load_state_dict(torch.load(self.model_path))

    def alter_obs_and_actions(self, obs, actions):

        self.modified_obs[:, [1, 2, 4, 5, 7, 8, 10, 11]] = obs[:, :8]
        self.modified_actions[:, [1, 2, 4, 5]] = actions[:, :4]
        mod_obs = self.modified_obs
        mod_act = self.modified_actions
        return mod_obs, mod_act

    def modified_new_obs(self, last_obs, action, new_obs):
        last_obs, action = self.alter_obs_and_actions(last_obs, action)
        new_obs, _ = self.alter_obs_and_actions(new_obs, action)
        # TODO: Check the dimensions properly
        input_tensor = torch.cat((last_obs, action, new_obs), 0).unsqueeze(0)

        lstm_model = self.load_model()
        with torch.no_grad():
            diff = lstm_model.forward(input_tensor)
        pred_obs = diff + input_tensor[:, :, 18:] # TODO: Check if this slicing is correct or not

        return pred_obs

    @staticmethod
    def revert_obs(pred_obs, obs):
        obs[:, :8] = pred_obs[:, [1, 2, 4, 5, 7, 8, 10, 11]]
        return obs

    def step(self, action): # TODO: Is this correct?
        obs = self.unwrapped._get_obs()
        new_obs, reward, done, info = self.step(action)
        pred = self.modified_new_obs(obs, action, new_obs)
        pred_obs = self.revert_obs(pred, new_obs)
        # TODO: Check the outputs

        return pred_obs, reward, done, info

    def reset(self, forced=False):
        super(ErgoReacherAugmented, self).reset()

    def render(self, mode='human', close=False):
        super(ErgoReacherAugmented, self).render()
    # TODO: Include the necessary methods if any










