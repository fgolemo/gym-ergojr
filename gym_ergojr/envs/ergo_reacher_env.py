import time
import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from gym_ergojr.sim.objects import Ball
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.utils.math import RandomPointInHalfSphere
from gym_ergojr.utils.pybullet import DistanceBetweenObjects

GOAL_REACHED_DISTANCE = -0.016  # distance between robot tip and goal under which the task is considered solved
RADIUS = 0.2022
DIA = 2 * RADIUS
RESET_EVERY = 5  # for the gripper


class ErgoReacherEnv(gym.Env):

    def __init__(self,
                 headless=True,
                 simple=False,
                 backlash=False,
                 max_force=1,
                 max_vel=18,
                 goal_halfsphere=False,
                 multi_goal=False,
                 goals=3,
                 terminates=True,
                 gripper=False):
        self.simple = simple
        self.backlash = backlash
        self.max_force = max_force
        self.max_vel = max_vel
        self.multigoal = multi_goal
        self.n_goals = goals
        self.gripper = gripper
        self.terminates = terminates

        self.goals_done = 0
        self.is_initialized = False
        self.robot = SingleRobot(debug=not headless, backlash=backlash)
        self.ball = Ball()
        self.rhis = RandomPointInHalfSphere(
            0.0,
            0.0369,
            0.0437,
            radius=RADIUS,
            height=0.2610,
            min_dist=0.1,
            halfsphere=goal_halfsphere)
        self.goal = None
        self.goal_positions = []
        self.goal_distances = []
        self.dist = DistanceBetweenObjects(
            bodyA=self.robot.id, bodyB=self.ball.id, linkA=13, linkB=1)
        self.episodes = 0  # used for resetting the sim every so often
        self.restart_every_n_episodes = 1000

        self.metadata = {'render.modes': ['human']}

        if not simple and not gripper:  # default
            # observation = 6 joints + 6 velocities + 3 coordinates for target
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(6 + 6 + 3,), dtype=np.float32)  #
            # action = 6 joint angles
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(6,), dtype=np.float32)  #

        elif not gripper:  # simple
            # observation = 4 joints + 4 velocities + 2 coordinates for target
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(4 + 4 + 2,), dtype=np.float32)  #
            # action = 4 joint angles
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32)  #
        else:  # gripper
            # observation = 3 joints + 3 velocities + 2 coordinates for target
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(3 + 3 + 2,), dtype=np.float32)  #
            # action = 3 joint angles, [-,1,2,-,4,-]
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float32)  #

        super().__init__()

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        if self.simple or self.gripper:
            action_ = np.zeros(6, np.float32)
            if self.simple:
                action_[[1, 2, 4, 5]] = action
            if self.gripper:
                action_[[1, 2, 4]] = action
            action = action_

        self.robot.act2(action, max_force=self.max_force, max_vel=self.max_vel)
        self.robot.step()

        reward, done, dist = self._getReward()

        obs = self._get_obs()

        if not self.terminates:
            done = False

        return obs, reward, done, {"distance": dist}

    def _getReward(self):
        done = False

        reward = self.dist.query()
        distance = reward.copy()
        if not self.multigoal:  # this is the normal mode
            reward *= -1  # the reward is the inverse distance
            if reward > GOAL_REACHED_DISTANCE:  # this is a bit arbitrary, but works well
                self.goals_done += 1
                done = True
                reward = 1
        else:
            dirty = False  # in case we _just_ hit the goal
            if -reward > GOAL_REACHED_DISTANCE:
                self.goals_done += 1
                if self.goals_done == self.n_goals:
                    done = True
                else:
                    robot_state = self._get_obs()[:8]
                    self.move_ball()
                    self._set_state(
                        robot_state)  # move robot back after ball has movedÒ
                    self.robot.step()
                    dirty = True

            if done or self.goals_done == self.n_goals:
                reward = 1
            else:
                # reward is distance to current target + sum of all other distances divided by total distance
                if dirty:
                    # take it off before the reward calc
                    self.goals_done -= 1

                reward = 1 + (-(reward + sum(
                    self.goal_distances[:-(self.goals_done + 1)])) /
                              sum(self.goal_distances))
                if dirty:
                    # add it back after the reward cald
                    self.goals_done += 1
                    dirty = False

        if self.gripper:
            reward *= 10
            if self.goals_done == RESET_EVERY:
                self.goals_done = 0
                self.reset(True)
            done = False

            # normalize - [-1,1] range:
            # reward = reward * 2 - 1

        return reward, done, distance

    def _setDist(self):
        self.dist.bodyA = self.robot.id
        self.dist.bodyB = self.ball.id

    def move_ball(self):
        if not self.multigoal or len(self.goal_positions) == 0:
            if self.simple or self.gripper:
                self.goal = self.rhis.sampleSimplePoint()
            else:
                self.goal = self.rhis.samplePoint()
        else:
            self.goal = self.goal_positions.pop()

        self.dist.goal = self.goal
        self.ball.changePos(self.goal, 4)

        for _ in range(25):
            self.robot.step()  # we need this to move the ball

    def reset(self, forced=False):
        self.goals_done = 0

        if self.multigoal:
            # sample N goals, calculate total reward as distance between them. Add distances to list. Subtract list elements on rew calculation
            self.goal_distances = []
            self.goal_positions = []
            for goal_idx in range(self.n_goals):
                if self.simple or self.gripper:
                    point = self.rhis.sampleSimplePoint()
                else:
                    point = self.rhis.samplePoint()
                self.goal_positions.append(point)
            for goal_idx in range(self.n_goals - 1):
                dist = np.linalg.norm(self.goal_positions[goal_idx] -
                                      self.goal_positions[goal_idx + 1])
                self.goal_distances.append(dist)

        self.episodes += 1
        if self.episodes >= self.restart_every_n_episodes:
            self.robot.hard_reset()  # this always has to go first
            self.ball.hard_reset()
            self._setDist()
            self.episodes = 0

        if self.is_initialized:
            robot_state = self._get_state()

        self.move_ball()

        if self.gripper and self.is_initialized:
            self._set_state(
                robot_state[:6])  # move robot back after ball has movedÒ
            self.robot.step()

        if forced or not self.gripper:  # if it's the gripper
            qpos = np.random.uniform(low=-0.2, high=0.2, size=6)

            if self.simple:
                qpos[[0, 3]] = 0

            self.robot.reset()
            self.robot.set(np.hstack((qpos, [0] * 6)))
            self.robot.act2(np.hstack((qpos)))
            self.robot.step()

        # add starting distance
        if self.multigoal:
            self.goal_distances.append(self.dist.query())

        self.is_initialized = True

        return self._get_obs()

    def _get_obs(self):
        obs = np.hstack([self.robot.observe(), self.rhis.normalize(self.goal)])
        if self.simple:
            obs = obs[[1, 2, 4, 5, 7, 8, 10, 11, 13, 14]]
        if self.gripper:
            obs = obs[[1, 2, 4, 7, 8, 10, 13, 14]]
        return obs

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.robot.close()

    def _get_state(self):
        return self.robot.observe()

    def _set_state(self, posvel):
        if self.simple or self.gripper:
            new_state = np.zeros((12), dtype=np.float32)
            if self.simple:
                new_state[[1, 2, 4, 5, 7, 8, 10, 11]] = posvel
            if self.gripper:
                new_state[[1, 2, 4, 7, 8, 10]] = posvel
        else:
            new_state = np.array(posvel)
        self.robot.set(new_state)


if __name__ == '__main__':
    import gym
    import gym_ergojr
    import time

    # MODE = "manual"
    env = gym.make("ErgoReacher-Graphical-Simple-v1")

    MODE = "manual"
    # env = gym.make("ErgoReacher-Graphical-Simple-Halfdisk-v1")
    # env = gym.make("ErgoReacher-Graphical-Gripper-MobileGoal-v1")

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
            # obs, rew, done, misc = env.step([17/90,-29/90,-33/90,-61/90])

            if MODE == "manual":
                print("act {}, obs {}, rew {}, done {}".format(
                    action, obs, rew, done))
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
                # env.unwrapped.goal = np.array([0., 0.01266761, 0.21479595])
                # env.unwrapped.dist.goal = np.array([0., 0.01266761, 0.21479595])
                # env.unwrapped.ball.changePos(env.unwrapped.goal, 4)
                break
