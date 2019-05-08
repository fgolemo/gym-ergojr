import time
import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from gym_ergojr.sim.abstract_robot import PusherRobot
from gym_ergojr.sim.objects import Puck

GOAL_REACHED_DISTANCE = 0.01
RESTART_EVERY_N_EPISODES = 1000


class ErgoPusherEnv(gym.Env):

    def __init__(self, headless=False):

        self.goals_done = 0
        self.is_initialized = False

        self.robot = PusherRobot(debug=not headless)
        self.puck = Puck()

        self.episodes = 0  # used for resetting the sim every so often

        self.metadata = {'render.modes': ['human']}

        # observation = 3 joints + 3 velocities + 2 coordinates for target
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3 + 3 + 2,), dtype=np.float32)  #

        # action = 3 joint angles
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32)  #

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        self.robot.act(action)
        self.robot.step()

        reward, done, dist = self._getReward()

        obs = self._get_obs()
        return obs, reward, done, {"distance": dist}

    def _getReward(self):
        done = False

        reward = self.puck.dbo.query()
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
            self.puck.hard_reset()
            self.episodes = 0
            self.is_initialized = True
        else:
            self.puck.reset()

        qpos = self.robot.rest_pos.copy()
        qpos[:3] += np.random.uniform(low=-0.1, high=0.1, size=3)

        self.robot.set(qpos)
        self.robot.act(qpos[:3])
        self.robot.step()

        return self._get_obs()

    def _get_obs(self):
        obs = np.hstack([self.robot.observe(), self.puck.dbo.goal[:2]])
        return obs

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.robot.close()

    def _get_state(self):
        return self.robot.observe()


if __name__ == '__main__':
    import gym
    import gym_ergojr
    import time

    env = gym.make("ErgoPusher-Graphical-v1")
    MODE = "manual"
    r = range(100)

    # env = gym.make("ErgoPusher-Headless-v1")
    # MODE = "timings"
    # r = tqdm(range(10000))

    env.reset()

    timings = []
    ep_count = 0

    start = time.time()

    for _ in r:
        while True:
            action = env.action_space.sample()
            obs, rew, done, misc = env.step(action)

            if MODE == "manual":
                print("act {}, obs {}, rew {}, done {}".format(
                    action, obs, rew, done))
                time.sleep(0.01)

            if MODE == "timings":
                ep_count += 1
                if ep_count >= 10000:
                    diff = time.time() - start
                    tqdm.write("avg. fps: {}".format(
                        np.around(10000 / diff, 3)))
                    np.savez("timings.npz", time=np.around(10000 / diff, 3))
                    ep_count = 0
                    start = time.time()

            if done:
                env.reset()
                break
