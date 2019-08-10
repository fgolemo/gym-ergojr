import numpy as np
import torch
from gym_ergojr.envs.ergo_reacher_env import ErgoReacherEnv
from gym_ergojr.models.networks import ReacherNetV1


class ErgoReacherAugmentedEnv(ErgoReacherEnv):

    def __init__(self,
                 headless=False,
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
        self.model = ReacherNetV1(
            n_input_state_sim=12,
            n_input_state_real=12,
            n_input_actions=6,
            nodes=self.hidden_layers,
            layers=self.lstm_layers)

        self.modified_obs = torch.zeros(1, 12).float()
        self.modified_actions = torch.zeros(1, 6).float()
        super(ErgoReacherAugmentedEnv, self).__init__()
        ErgoReacherEnv.__init__(
            self,
            headless=headless,
            simple=simple,
            backlash=backlash,
            max_force=max_force,
            max_vel=max_vel,
            goal_halfsphere=goal_halfsphere,
            multi_goal=multi_goal,
            goals=goals,
            gripper=gripper)

        self.model_path = '../trained_lstms/ergoreacher-exp1-h128-l3-v01-e5.pth'
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(
            torch.load(self.model_path, map_location='cpu'))

    def obs2lstm(self, obs):
        self.modified_obs[:, [1, 2, 4, 5, 7, 8, 10, 11]] = obs[:, :8]

        # this is important, otherwise it's a reference
        return self.modified_obs.clone()

    def augment(self, last_obs, action, new_obs):
        last_obs = self.obs2lstm(last_obs)
        new_obs = self.obs2lstm(new_obs)
        self.modified_actions[:, [1, 2, 4, 5]] = action[:, :4]
        action = self.modified_actions.clone()

        input_tensor = torch.cat((last_obs, action, new_obs), 1).unsqueeze(0)

        with torch.no_grad():
            diff = self.model.forward(input_tensor)

        return diff.squeeze(0)

    # Flo: I don't think we need an extra function call for this, when it can be done in one line
    # @staticmethod
    # def lstm2obs(diff, obs):
    #     obs[:, :8] += diff[:, [1, 2, 4, 5, 7, 8, 10, 11]]
    #     return obs

    @staticmethod
    def convert_to_tensor(numpy_array):
        return torch.FloatTensor(np.expand_dims(numpy_array, 0))

    def step(self, action):
        obs = super()._get_obs()
        new_obs, _, _, _ = super().step(action)

        obs = self.convert_to_tensor(obs)
        action = self.convert_to_tensor(action)
        new_obs = self.convert_to_tensor(new_obs)

        obs_diff = self.augment(obs, action, new_obs)

        corrected_obs = (new_obs[:, :8] +
                         obs_diff[:, [1, 2, 4, 5, 7, 8, 10, 11]]).numpy()

        super()._set_state(corrected_obs[:, :8])

        reward, done, info = super()._getReward()
        return corrected_obs, reward, done, info

    def reset(self, forced=False):
        super().reset()
        self.model.zero_hidden()  # !important
        self.model.hidden = (self.model.hidden[0].detach(),
                             self.model.hidden[1].detach())

    def render(self, mode='human', close=False):
        super().render()


if __name__ == '__main__':
    import gym
    import time
    env = gym.make("ErgoReacherAugmented-Headless-MultiGoal-Halfdisk-v1")

    env.reset()

    done = False

    while not done:
        action = [-1, 1, -1, 1]
        obs, rew, done, misc = env.step(action)
        time.sleep(.1)
        # quit()
