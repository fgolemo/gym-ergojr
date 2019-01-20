import time
import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from gym_ergojr.sim.ball import Ball
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.utils.math import RandomPointInHalfSphere
from gym_ergojr.utils.pybullet import DistanceBetweenObjects


class ErgoReacherHeavyEnv(gym.Env):
    def __init__(self, headless=False, simple=False, max_force=1000, max_vel=100, goal_halfsphere=False, backlash=.1):
        self.simple = simple
        self.max_force = max_force
        self.max_vel = max_vel

        self.robot = SingleRobot(debug=not headless, heavy=True, new_backlash=backlash, silent=True)
        self.ball = Ball(1)
        self.rhis = RandomPointInHalfSphere(0.0, 3.69, 4.37,
                                            radius=20.22, height=26.10,
                                            min_dist=10., halfsphere=goal_halfsphere)
        self.goal = None
        self.dist = DistanceBetweenObjects(bodyA=self.robot.id, bodyB=self.ball.id,
                                           linkA=19, linkB=1)
        self.episodes = 0  # used for resetting the sim every so often
        self.restart_every_n_episodes = 1000

        self.force_urdf_reload = False

        self.metadata = {
            'render.modes': ['human']
        }

        if not simple:
            # observation = 6 joints + 6 velocities + 3 coordinates for target
            self.observation_space = spaces.Box(low=-1, high=1, shape=(6 + 6 + 3,), dtype=np.float32)  #
            # action = 6 joint angles
            self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  #

        else:
            # observation = 4 joints + 4 velocities + 2 coordinates for target
            self.observation_space = spaces.Box(low=-1, high=1, shape=(4 + 4 + 2,), dtype=np.float32)  #
            # action = 4 joint angles
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  #

        super().__init__()

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        if self.simple:
            action_ = np.zeros(6, np.float32)
            action_[[1, 2, 4, 5]] = action
            action = action_

        self.robot.act2(action, max_force=self.max_force, max_vel=self.max_vel)
        self.robot.step()
        self.robot.step()
        self.robot.step()

        reward, done = self._getReward()

        obs = self._get_obs()
        return obs, reward, done, {}

    def _getReward(self):
        done = False

        reward = self.dist.query()
        reward *= -1  # the reward is the inverse distance

        if reward > -1.6:  # this is a bit arbitrary, but works well
            done = True
            reward = 1

        return reward, done

    def _setDist(self):
        self.dist.bodyA = self.robot.id
        self.dist.bodyB = self.ball.id

    def update_backlash(self, new_val):
        self.robot.new_backlash = new_val
        self.force_urdf_reload = True
        # and now on the next self.reset() the new modified URDF will be loaded

    def reset(self):
        self.episodes += 1
        if self.force_urdf_reload or self.episodes >= self.restart_every_n_episodes:
            self.robot.hard_reset()  # this always has to go first
            self.ball.hard_reset()
            self._setDist()
            self.episodes = 0
            self.force_urdf_reload = False

        if self.simple:
            self.goal = self.rhis.sampleSimplePoint()
        else:
            self.goal = self.rhis.samplePoint()

        print(self.goal)
        self.dist.goal = self.goal

        self.ball.changePos(self.goal)
        for _ in range(20):
            self.robot.step()  # we need this to move the ball

        qpos = np.random.uniform(low=-0.2, high=0.2, size=6)

        if self.simple:
            qpos[[0, 3]] = 0

        self.robot.reset()
        self.robot.set(np.hstack((qpos, [0] * 6)))
        self.robot.act2(np.hstack((qpos)))
        self.robot.step()

        return self._get_obs()

    def _get_obs(self):
        obs = np.hstack([
            self.robot.observe(),
            self.rhis.normalize(self.goal)
        ])
        if self.simple:
            obs = obs[[1, 2, 4, 5, 7, 8, 10, 11, 13, 14]]
        return obs

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.robot.close()

    def _get_state(self):
        return self.robot.observe()

    def _set_state(self, posvel):
        if self.simple:
            new_state = np.zeros((12), dtype=np.float32)
            new_state[[1, 2, 4, 5, 7, 8, 10, 11]] = posvel
        else:
            new_state = np.array(posvel)
        self.robot.set(new_state)


if __name__ == '__main__':
    import gym
    import gym_ergojr
    import time

    MODE = "manual"
    env = gym.make("ErgoReacher-Graphical-Simple-Halfdisk-Heavy-v1")

    # MODE = "timings"
    # env = gym.make("ErgoReacher-Headless-Simple-Halfdisk-Heavy-v1")

    env.reset()

    timings = []
    ep_count = 0

    start = time.time()

    if MODE == "manual":
        r = range(100)
    else:
        r = tqdm(range(10000))

    for _ in r:
        while True:
            action = env.action_space.sample()
            obs, rew, done, misc = env.step(action)

            if MODE == "manual":
                print("act {}, obs {}, rew {}, done {}".format(
                    action,
                    obs,
                    rew,
                    done
                ))
                time.sleep(0.01)

            if MODE == "timings":
                ep_count += 1
                if ep_count >= 10000:
                    diff = time.time() - start
                    print("avg. fps: {}".format(np.around(10000 / diff, 3)))
                    np.savez("timings.npz", time=np.around(10000 / diff, 3))
                    ep_count = 0
                    start = time.time()

            if done:
                env.reset()
                break
