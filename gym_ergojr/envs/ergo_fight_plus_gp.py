import os

import gym
import numpy as np
from sklearn.externals import joblib


class ErgoFightPlusGPWrapper(gym.Wrapper):
    def __init__(self, env, model="2_1000"):
        super().__init__(env)
        self.env = env
        # self.scaling = scaling
        self.scaling = 1

        modelFile = "../trained_lstms/gp{}.pkl".format(model)
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), modelFile)
        self.gp = joblib.load(full_path)
        self.step_counter = 0

    def prep(self, sim_t2, real_t1, action):
        return np.expand_dims(np.hstack((sim_t2, real_t1, action)), axis=0)


    def step(self, action):
        self.step_counter += 1
        obs_real_t1 = self.unwrapped._self_observe()

        obs_sim_t2, _, _, info = self.unwrapped.step(action, dry_run=True)
        variable = self.prep(obs_sim_t2[:12].copy(), obs_real_t1[:12].copy(), np.array(action).copy())

        obs_real_t2_delta = self.gp.predict(variable)[0]

        obs_real_t2 = obs_sim_t2[:12].copy() + self.scaling * obs_real_t2_delta

        new_obs = self.unwrapped.set_state(obs_real_t2)

        # print("real t1:", obs_real_t1[:12].round(2))
        # print("sim_ t2:", obs_sim_t2[:12].round(2))
        # print("action_:", np.around(action,2))
        # print("delta__:", obs_real_t2_delta)
        # print("real t2:", obs_real_t2[:12].round(2))
        # print("===")

        done = False
        if self.step_counter >= self.env._max_episode_steps:
            _ = self.reset()  # automatically reset the env
            done = True  # this is nasty but IDK how else to do it

        return new_obs, self.unwrapped._getReward(), done, info

    def reset(self):
        self.step_counter = 0
        return self.env.reset()

    def set_state(self, state):
        return self.unwrapped.set_state(state)


def ErgoFightPlusGPEnv(base_env_id, model="2_1000"):
    return ErgoFightPlusGPWrapper(gym.make(base_env_id), model=model)


if __name__ == '__main__':
    import gym_ergojr
    import time

    env = gym.make("ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-PlusGP-v0")

    env.reset()

    # for episode in range(10):
    for step in range(10):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        # time.sleep(0.01)
        print(step, rew, done, info)
    # env.reset()
