# import matplotlib
# print(matplotlib.rcsetup.interactive_bk) # ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo']

import time
import gym
import numpy as np
from gym import spaces

from gym_ergojr.sim.objects import Cube
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.utils.pybullet import Cam
import matplotlib.pyplot as plt

GOAL_REACHED_DISTANCE = -0.007  # distance between robot tip and goal under which the task is considered solved
RESTART_EVERY_N_EPISODES = 5  # for the gripper


class ErgoGripperEnv(gym.Env):

    def __init__(self, headless=False, multi_goal=False, goals=3):
        # self.multigoal = multi_goal # unused
        # self.n_goals = goals # unused

        self.goals_done = 0
        self.is_initialized = False

        self.robot = SingleRobot(
            robot_model="ergojr-gripper",
            debug=not headless,
            gripper=True,
            reset=False)
        self.cube = None  # after reset
        self.cam = Cam(pos=[.3, .05, .1], look_at=[0, .07, 0.1])
        self.plot = None

        self.episodes = 0  # used for resetting the sim every so often

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        # observation = (img, 6 joints + 6 velocities + 3 cube position)
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=255, shape=(300, 400, 3), dtype=np.uint8),
            spaces.Box(
                low=-1, high=1, shape=(3 + 3 + 2 + 2,), dtype=np.float32)
        ])

        # action = 6 joint angles
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32)  #

        super().__init__()

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        self.robot.act2(action)
        self.robot.step()

        reward, done, dist = self._getReward()

        obs = self._get_obs()
        return obs, reward, done, {"distance": dist}

    def _getReward(self):
        done = False

        reward = self.cube.dbo.query()
        distance = reward.copy()

        reward *= -1  # the reward is the inverse distance
        if distance < GOAL_REACHED_DISTANCE:  # this is a bit arbitrary, but works well
            done = True
            reward = 1

        return reward, done, distance

    def reset(self, forced=False):
        self.episodes += 1
        if self.episodes >= RESTART_EVERY_N_EPISODES or forced or not self.is_initialized:
            self.robot.hard_reset()  # this always has to go first
            self.cube = Cube(self.robot.id)
            self.cube.reset()
            self.episodes = 0
            self.is_initialized = True
        else:
            self.cube.reset()

        qpos = self.robot.rest_pos.copy()
        qpos += np.random.uniform(low=-0.2, high=0.2, size=6)

        qposvel = np.zeros(12, dtype=np.float32)
        qposvel[:6] = qpos

        self.robot.set(qposvel)
        self.robot.act2(qposvel[:6])
        self.robot.step()

        return self._get_obs()

    def _get_obs(self):
        obs = np.hstack([self.robot.observe(), self.cube.normalize_cube()])
        img = self.cam.snap()
        return img, obs

    def render(self, mode='human', close=False):
        img = self.cam.snap()
        if mode == "human":
            if self.plot is None:
                plt.ion()
                self.plot_container = plt.imshow(
                    img, interpolation='none', animated=True, label="live feed")
                self.plot = plt.gca()

            self.plot_container.set_data(img)
            self.plot.plot([0])
            plt.pause(0.001)
        else:
            return img

    def close(self):
        self.robot.close()

    def _get_state(self):
        return self.robot.observe()


if __name__ == '__main__':
    import gym
    import gym_ergojr

    env = gym.make("ErgoGripper-Headless-v1")

    for i in range(5):
        env.reset()
        done = False
        print("\n\n=== RESET ===\n\n")

        while not done:
            action = env.action_space.sample()
            (img, obs), rew, done, misc = env.step(action)
            env.render("human")
            print(img.shape, obs, rew, done, misc)
