import os
import time
import gym
import numpy as np
from gym import spaces
import logging
from tqdm import tqdm

from gym_ergojr.sim.double_robot import DoubleRobot

REST_POS = [0] * 6
RANDOM_NOISE = [  # in degrees
    (-90, 90),
    (-30, 30),
    (-30, 30),
    (-45, 45),
    (-30, 30),
    (-30, 30)
]
MOVE_EVERY_N_STEPS = 250


class ErgoFightStaticEnv(gym.Env):
    def __init__(self, headless=True, scaling=1):
        self.headless = headless
        self.scaling = scaling

        self.step_in_episode = 0
        self.randomPos = {0: [], 1: []}

        self.robots = DoubleRobot(debug=not headless)

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        # 6 own joint pos, 6 own joint vel, 6 enemy joint pos, 6 enemy joint vel
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6 + 6 + 6 + 6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

    def seed(self, seed=None):
        np.random.seed(seed)

    def _restPos(self):
        self.done = False
        self.robots.rest()
        self.randomize(robot=1, scaling=self.scaling)

    def randomize(self, robot=1, scaling=1.0):
        self.randomPos[robot] = []
        for i in range(6):
            new_pos = scaling * np.random.randint(
                low=RANDOM_NOISE[i][0],
                high=RANDOM_NOISE[i][1],
                size=1)[0]
            new_pos = (new_pos + 90) / 180
            new_pos = new_pos * 2 - 1
            self.randomPos[robot].append(new_pos)

        self.robots.set(self.randomPos[robot] + [0] * 6, robot)
        # just resetting robot is not enough,
        # also need to tell motors where to stay
        self.robots.act2(self.randomPos[robot], robot)

    def reset(self):
        self.step_in_episode = 0
        self._restPos()

        return self._self_observe()

    def _getReward(self):
        reward = 0
        
        if self.step_in_episode > 0:
            hits = self.robots.get_hits(links=(14, 14))
            if len(hits) > 0:
                reward = 1
                self._restPos()

        return reward

    def _self_observe(self):
        self.observation = self.robots.observe_both()
        return self.observation

    def step(self, actions):
        self.step_in_episode += 1

        robot = 0

        self.robots.act2(actions, robot_id=robot)

        self.robots.step()

        if self.step_in_episode % MOVE_EVERY_N_STEPS == 0:
            self.randomize(1, scaling=self.scaling)

        return self._self_observe(), self._getReward(), self.done, {}

    def close(self):
        self.robots.close()

    def render(self, mode='human', close=False):
        # This intentionally does nothing and is only here for wrapper functions.
        # if you want graphical output, use the environments
        # "ErgoBallThrowAirtime-Graphical-Normalized-v0"
        # or
        # "ErgoBallThrowAirtime-Graphical-v0"
        # ... not the ones with "...-Headless-..."
        pass
