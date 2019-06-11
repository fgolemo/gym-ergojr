import time
from random import random, sample

import gym
import numpy as np
from gym import spaces

from gym_ergojr.sim.objects import Cube
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.utils.pybullet import Cam
import matplotlib.pyplot as plt

GOAL_REACHED_DISTANCE = 0.04  # distance between robot tip and goal under which the task is considered solved
RESTART_EVERY_N_EPISODES = 100  # for the gripper
FRAME_SKIP = 3


class ErgoGripperEnv(gym.Env):

    def __init__(self, headless=False, cube_spawn="linear"):
        assert cube_spawn in ["linear", "square"]
        self.cube_spawn = cube_spawn

        self.goals_done = 0
        self.is_initialized = False

        self.robot = SingleRobot(
            robot_model="ergojr-gripper",
            debug=not headless,
            gripper=True,
            reset=False,
            frequency=60)

        # Cube the robot must reach, created in reset method
        self.cube = None

        self.cam = Cam(
            pos=[0.25, .25, 0.1],
            look_at=[0.00, .10, 0.1],
            width=64,
            height=64,
            fov=60)

        self.plot = None

        # Episode count, used for resetting the PyBullet sim every so often
        self.episodes = 0

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        # observation = (img, 6 joints + 6 velocities + 3 cube position)
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            spaces.Box(
                low=-1, high=1, shape=(3 + 3 + 2 + 2,), dtype=np.float32)
        ])

        # 6 joint angles in [-1, 1]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32)

        super().__init__()

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        self.robot.act2(action, max_vel=.8, max_force=.8)

        for _ in range(FRAME_SKIP):
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
            if self.cube is not None:
                self.cube.cleanup()

            self.robot.hard_reset()  # this always has to go first

            self.cube = Cube(self.robot.id, self.cube_spawn)
            self.cube.reset()
            self.episodes = 0
            self.is_initialized = True
        else:
            self.cube.reset()

        qpos = self.robot.rest_pos.copy()
        qpos += np.random.uniform(low=-0.2, high=0.2, size=6)

        qposvel = np.zeros(12, dtype=np.float32)
        qposvel[:6] = qpos

        for _ in range(20):
            # to stabilize cube at lower framerates
            self.robot.step()

        self.robot.set(qposvel)
        self.robot.act2(qposvel[:6])
        self.robot.step()

        return self._get_obs()

    def _get_obs(self):
        obs = np.hstack([self.robot.observe(), self.cube.normalize_cube()])
        img = self.cam.snap()
        img = (img * 255).astype(np.uint8)
        return img, obs

    def render(self, mode='human', close=False):
        if mode == "human":
            img = self.cam.snap()
            if self.plot is None:
                plt.ion()
                self.plot_container = plt.imshow(
                    img, interpolation='none', animated=True, label="live feed")
                self.plot = plt.gca()

            self.plot_container.set_data(img)
            self.plot.plot([0])
            plt.pause(0.001)
        else:
            return self._get_obs()[0]

    def close(self):
        self.robot.close()

    def _get_state(self):
        return self.robot.observe()


class OnlyImageWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # only the image part
        self.observation_space = self.observation_space[0]

    def observation(self, observation):
        return observation[0]  # only the img


if __name__ == '__main__':
    import gym_ergojr
    env = OnlyImageWrapper(gym.make("ErgoGripper-Square-Graphical-v1"))
    print(env.observation_space)
    env.reset()
    env.render("human")
    actions = [[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 1, -1, 0, 1, 0],
               [0, -1, 1, 0, -1, 0]]
    for i in range(1000):
        if i % 50 == 0:
            action = sample(actions, 1)[0]

        env.step(action)
        env.render("human")

        if i % 50 == 0:
            env.reset()
