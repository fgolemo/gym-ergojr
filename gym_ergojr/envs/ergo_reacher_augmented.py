import numpy as np
import torch
from gym_ergojr.envs.ergo_reacher_env import ErgoReacherEnv
from gym_ergojr.models.networks import LstmNetRealv1


class ErgoReacherAugmentedEnv(ErgoReacherEnv):
    def __init__(self, headless=False,
               simple=False,
               backlash=False,
               max_force=1,
               max_vel=18,
               goal_halfsphere=False,
               multi_goal=False,
               goals=3,
               gripper=False):
        self.hidden_layers = 128
        self.lstm_layers = 3
        self.model = LstmNetRealv1(
                n_input_state_sim=12,
                n_input_state_real=12,
                n_input_actions=6,
                nodes=self.hidden_layers,
                layers=self.lstm_layers)
        self.model_path = '/home/sharath/neural-augmented-simulator/nas/data/model-exp1-h128-l3-v01-e5.pth'
        self.modified_obs = torch.zeros(1, 12).float()
        self.modified_actions = torch.zeros(1, 6).float()
        super(ErgoReacherAugmentedEnv, self).__init__()
        ErgoReacherEnv.__init__(self, headless=headless,
               simple=simple,
               backlash=backlash,
               max_force=max_force,
               max_vel=max_vel,
               goal_halfsphere=goal_halfsphere,
               multi_goal=multi_goal,
               goals=goals,
               gripper=gripper)


    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))

    def alter_obs_and_actions(self, obs, actions):
        self.modified_obs[:, [1, 2, 4, 5, 7, 8, 10, 11]] = obs[:, :8]

        self.modified_actions[:, [1, 2, 4, 5]] = actions[:, :4]
        mod_obs = self.modified_obs
        mod_act = self.modified_actions
        return mod_obs, mod_act

    def modified_new_obs(self, last_obs, action, new_obs):
        last_obs, action = self.alter_obs_and_actions(last_obs, action)
        new_obs, _ = self.alter_obs_and_actions(new_obs, action)
        input_tensor = torch.cat((last_obs, action, new_obs), 1).unsqueeze(0)

        self.load_model()
        with torch.no_grad():
            diff = self.model.forward(input_tensor)
        pred_obs = diff + input_tensor[:, :, 18:]

        return pred_obs.squeeze(0)

    @staticmethod
    def revert_obs(pred_obs, obs):
        obs[:, :8] = pred_obs[:, [1, 2, 4, 5, 7, 8, 10, 11]]

        return obs

    @staticmethod
    def convert_to_tensor(numpy_array):
        return torch.FloatTensor(np.expand_dims(numpy_array, 0))

    def step(self, action):
        obs = super()._get_obs()
        new_obs, _, _, _ = super().step(action)
        obs = self.convert_to_tensor(obs)
        action = self.convert_to_tensor(action)
        new_obs = self.convert_to_tensor(new_obs)
        pred = self.modified_new_obs(obs, action, new_obs)
        pred_obs = self.revert_obs(pred, new_obs).numpy()
        super()._set_state(pred_obs[:, :8])
        print(super()._set_state(pred_obs[:, :8]))
        reward, done, info = super()._getReward()
        return pred_obs, reward, done, info

    def reset(self, forced=False):
        super().reset()

    def render(self, mode='human', close=False):
        super().render()










