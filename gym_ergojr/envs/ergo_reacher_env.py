import time

import gym
import numpy as np
from gym import spaces

from gym_ergojr.sim.ball import Ball
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.utils.math import RandomPointInHalfSphere
from gym_ergojr.utils.pybullet import DistanceBetweenObjects


class ErgoReacherEnv(gym.Env):
    def __init__(self, headless=False):
        self.robot = SingleRobot(debug=not headless)
        self.ball = Ball()
        self.rhis = RandomPointInHalfSphere(0.0, 0.0369, 0.0437,
                                            radius=0.2022, height=0.2610,
                                            min_dist=0.0477)
        self.goal = None
        self.dist = DistanceBetweenObjects(bodyA=self.robot.id, bodyB=self.ball.id,
                                           linkA=13, linkB=-1)

        self.metadata = {
            'render.modes': ['human']
        }

        # observation = 6 joints + 3 coordinates for target
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6 + 3,)) # , dtype=np.float32

        # action = 6 joint angles
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,)) # , dtype=np.float32

        super().__init__()

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        self.robot.act2(action)
        self.robot.step()
        done = False

        reward = self.dist.query()
        if reward is None: # then summin is wrong
            reward = -99
        else:
            reward *= -1 # the reward is the inverse distance

        if reward > -0.01:
            done = True

        obs = self._get_obs()
        return obs, reward, done, {}

    def reset(self):
        qpos = np.random.uniform(low=-0.1, high=0.1, size=6)
        self.robot.reset()
        self.robot.set(np.hstack((qpos, [0] * 6)))
        self.robot.act2(np.hstack((qpos)))

        self.goal = self.rhis.samplePoint()
        self.ball.changePos(self.goal)
        for _ in range(15):
            self.robot.step()  # we need this to move the ball

        return self._get_obs()

    def _get_obs(self):
        return np.hstack([
            self.robot.observe()[:6],
            self.goal
        ])

    def render(self):
        pass

    def close(self):
        self.robot.close()

if __name__ == '__main__':
    import gym
    import gym_ergojr

    env = gym.make("ErgoReacher-Graphical-v0")
    env.reset()

    for i in range(10):
        while True:
            action = env.action_space.sample()
            obs, rew, done, misc = env.step(action)

            print ("act {}, obs {}, rew {}, done {}".format(
                action,
                obs,
                rew,
                done
            ))

            time.sleep(0.01)

            if done:
                env.reset()
                break

