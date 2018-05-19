import os
import time
import gym
import numpy as np
from gym import spaces
import logging

from pytorch_a2c_ppo_acktr.inference import Inference
from skimage.transform import resize
from gym_ergojr.envs.constants import JOINT_LIMITS, BALL_STATES, RANDOM_NOISE, MOVE_EVERY_N_STEPS, JOINT_LIMITS_SPEED
from tqdm import tqdm
from vrepper.core import vrepper

logger = logging.getLogger(__name__)

REST_POS = [0, 0, 0, 0, 0, 0]
RANDOM_NOISE = [
    (-90, 90),
    (-30, 30),
    (-30, 30),
    (-45, 45),
    (-30, 30),
    (-30, 30)
]
INVULNERABILITY_AFTER_HIT = 3  # how many frames after a hit to reset
IMAGE_SIZE = (84, 84)


class ErgoFightStaticEnv(gym.Env):
    def __init__(self, headless=True, with_img=True,
                 only_img=False, fencing_mode=False, defence=False,
                 sword_only=False, fat=False, no_move=False, scaling=1, shield=False):
        self.headless = headless
        self.with_img = with_img
        self.only_img = only_img
        self.fencing_mode = fencing_mode
        self.defence = defence
        self.sword_only = sword_only
        self.fat = fat
        self.no_move = no_move
        self.scaling = scaling
        self.shield = shield

        self.step_in_episode = 0
        self.randomPos = {0:[],1:[]}

        if self.defence:
            # load up the inference model for the attacker
            self.inf = Inference("/home/florian/dev/pytorch-a2c-ppo-acktr/"
                                 "trained_models/ppo/"
                                 "ErgoFightStatic-Headless-Fencing-v0-180301140937.pt")

        self._startEnv(headless)

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        joint_boxes = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        if self.with_img:
            cam_image = spaces.Box(low=0, high=255, shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

            if self.only_img:
                self.observation_space = cam_image
            else:
                own_joints = spaces.Box(low=-1, high=1, shape=(6 + 6,))  # 6 joint pos, 6 joint vel
                self.observation_space = spaces.Tuple((cam_image, own_joints))
        else:
            # 6 own joint pos, 6 own joint vel, 6 enemy joint pos, 6 enemy joint vel
            all_joints = spaces.Box(low=-1, high=1, shape=(6 + 6 + 6 + 6,), dtype=np.float32)
            self.observation_space = all_joints

        self.action_space = joint_boxes

        self.diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]
        self.frames_after_hit = -1  # -1 means no recent hit, anything 0 or above means it's counting

    def seed(self, seed=None):
        np.random.seed(seed)

    def _startEnv(self, headless):
        self.venv = vrepper(headless=headless)
        self.venv.start()
        current_dir = os.path.dirname(os.path.realpath(__file__))

        scene = current_dir + '/../scenes/poppy_ergo_jr_fight_{}.ttt'
        if self.shield:
            file_to_load = "shield"
        elif self.sword_only:
            file_to_load = "sword_only_sword"
            if self.fat:
                file_to_load += "_fat"
        else:
            file_to_load = "sword1"

        scene = scene.format(file_to_load)

        self.venv.load_scene(scene)
        self.motors = ([], [])
        for robot_idx in range(2):
            for motor_idx in range(6):
                motor = self.venv.get_object_by_name('r{}m{}'.format(robot_idx + 1, motor_idx + 1), is_joint=True)
                self.motors[robot_idx].append(motor)
        collision_obj = "sword_hit"
        if self.fat:
            collision_obj += "_fat"
        self.sword_collision = self.venv.get_collision_object(collision_obj)
        self.cam = self.venv.get_object_by_name('cam', is_joint=False).handle
        # self.tip = self.frames_after_hit

    def _restPos(self):
        self.done = False
        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=True)

        for i, m in enumerate(self.motors[0]):
            m.set_position_target(REST_POS[i])

        self.randomize(robot=1, scaling=self.scaling)

        for _ in range(20):
            self.venv.step_blocking_simulation()

    def randomize(self, robot=1, scaling=1.0):
        self.randomPos[robot] = []
        for i in range(6):
            new_pos = REST_POS[i] + scaling * np.random.randint(
                low=RANDOM_NOISE[i][0],
                high=RANDOM_NOISE[i][1],
                size=1)[0]
            self.randomPos[robot].append(new_pos)
            self.motors[robot][i].set_position_target(new_pos)

    def reset(self):
        self.step_in_episode = 0
        self._restPos()
        self._self_observe()
        self.frames_after_hit = -1  # this enables hits / disables invulnerability frame
        return self.observation

    def _getReward(self):
        # The only way of getting reward is by hitting and releasing, hitting and releasing.
        # Just touching and holding doesn't work.
        reward = 0
        if self.sword_collision.is_colliding() and self.frames_after_hit == -1:
            reward = 1
            if not self.fencing_mode:
                self.frames_after_hit = 0
            else:
                self._restPos()  # if fencing mode then reset pos on each hit

        # the following bit is for making sure the robot doen't just hit repeatedly
        # ...so the invulnerability countdown only start when the collision is released
        else:  # if it's not hitting anything right now
            if self.frames_after_hit >= 0:
                self.frames_after_hit += 1
            if self.frames_after_hit >= INVULNERABILITY_AFTER_HIT:
                self.frames_after_hit = -1

        if self.defence:
            reward *= -1

        return reward

    def _get_robot_posvel(self, robot_id):
        pos = []
        vel = []
        for i, m in enumerate(self.motors[robot_id]):
            pos.append(m.get_joint_angle())
            vel.append(m.get_joint_velocity()[0])

        pos = self._normalize(pos)  # move pos into range [-1,1]
        vel = self._normalizeVel(vel)

        joint_vel = np.hstack((pos, vel)).astype('float32')
        return joint_vel

    def _self_observe(self):
        own_joint_vel = self._get_robot_posvel(0)
        if self.with_img:
            cam_image = self.venv.flip180(self.venv.get_image(self.cam))
            cam_image = resize(cam_image, IMAGE_SIZE)
            if self.only_img:
                self.observation = cam_image
            else:
                self.observation = (cam_image, own_joint_vel)
        else:
            enemy_joint_vel = self._get_robot_posvel(1)
            self.observation = np.hstack((own_joint_vel, enemy_joint_vel)).astype('float32')
        return self.observation

    def _gotoPos(self, pos, robot=0):
        for i, m in enumerate(self.motors[robot]):
            m.set_position_target(pos[i])

    def _forcePos(self, pos, robot=0):
        for i, m in enumerate(self.motors[robot]):
            m.force_position(pos[i])

    def _normalize(self, pos):
        out = []
        for i in range(6):
            shifted = (pos[i] - JOINT_LIMITS[i][0]) / self.diffs[i]  # now it's in [0,1]
            norm = shifted * 2 - 1
            out.append(norm)
        return out

    def _normalizeVel(self, vel):
        vel = np.array(vel)
        shifted = (vel + JOINT_LIMITS_SPEED) / (JOINT_LIMITS_SPEED * 2)
        norm = shifted * 2 - 1

        return norm

    def _denormalize(self, actions):
        out = []
        for i in range(6):
            shifted = (actions[i] + 1) / 2  # now it's within [0,1]
            denorm = shifted * self.diffs[i] + JOINT_LIMITS[i][0]
            out.append(denorm)
        return out

    def _denormalizeVel(self, vel):
        vel = np.array(vel)  # now it's in range [-1;1]
        shifted = (vel + 1) / 2  # now it's in range [0;1]
        denorm = (shifted * JOINT_LIMITS_SPEED * 2) - JOINT_LIMITS_SPEED
        return denorm

    def prep_actions(self, actions):
        actions = np.clip(actions, -1, 1)  # first make sure actions are normalized
        actions = self._denormalize(actions)  # then scale them to the actual joint angles
        return actions

    def set_state(self, pos, vel):
        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=True)

        for i, m in enumerate(self.motors[0]):
            m.set_position_target(pos[i])

        for i, m in enumerate(self.motors[1]):
            m.set_position_target(self.randomPos[1][i])

        for _ in range(30):
            self.venv.step_blocking_simulation()

        # TODO: velocity is currently not being set. Check if this is necessary.
        # TODO: this could be done via simxSetObjectFloatParameter (param 3000-3002)

        # params = self.venv.create_params() #OHNONONONON BAD
        # self.venv.call_script_function("resetDynamics", params)

    def step(self, actions, dont_terminate=False):
        self.step_in_episode += 1
        actions = self.prep_actions(actions)

        robot = 0

        attacker_action = []
        if self.defence:
            attacker_action = self.prep_actions(self.inf.get_action(self.observation))
            self._gotoPos(attacker_action, robot=0)
            robot = 1

        self._gotoPos(actions, robot=robot)
        self.venv.step_blocking_simulation()

        if not self.no_move:
            if (not self.sword_only and not self.defence and self.step_in_episode % 5 == 0) or \
                    (self.sword_only and self.step_in_episode % MOVE_EVERY_N_STEPS == 0):
                # print("randomizing, episode", self.step_in_episode)
                self.randomize(1, scaling=self.scaling)

        # observe again
        self._self_observe()

        rew = None
        if not dont_terminate:
            rew = self._getReward()

        return self.observation, rew, self.done, {"attacker": attacker_action}

    def close(self):
        self.venv.stop_simulation()
        self.venv.end()

    def render(self, mode='human', close=False):
        # This intentionally does nothing and is only here for wrapper functions.
        # if you want graphical output, use the environments
        # "ErgoBallThrowAirtime-Graphical-Normalized-v0"
        # or
        # "ErgoBallThrowAirtime-Graphical-v0"
        # ... not the ones with "...-Headless-..."
        pass


if __name__ == '__main__':
    import gym_ergojr
    import matplotlib.pyplot as plt


    def test_normal_mode():

        env = gym.make("ErgoFightStatic-Graphical-v0")

        plt.ion()
        img = np.random.uniform(0, 255, (256, 256, 3))
        plt_img = plt.imshow(img, interpolation='none', animated=True, label="blah")
        plt_ax = plt.gca()

        for k in range(3):
            observation = env.reset()
            print("init done")
            time.sleep(2)
            for i in range(30):
                if i % 5 == 0:
                    # action = env.action_space.sample() # this doesn't work
                    action = np.random.uniform(low=-1.0, high=1.0, size=(6))
                observation, reward, done, info = env.step(action)
                plt_img.set_data(observation[0])
                plt_ax.plot([0])
                plt.pause(0.001)  # I found this necessary - otherwise no visible img
                print(action, observation[0].shape, observation[1], reward, done)
                print(".")

        env.close()

        print('simulation ended. leaving in 5 seconds...')
        time.sleep(2)


    def test_fencing_defence():
        env = gym.make("ErgoFightStatic-Headless-Fencing-Defence-v0")

        env.reset()

        # att_actions = []
        for _ in range(100):
            act = env.action_space.sample()
            obs, rew, _, misc = env.step(act)
            # print (act, obs, rew)
            print(rew)
            # att_actions.append(misc["attacker"])

        ## PLOT ACTIONS

        # att_actions = np.array(att_actions)
        # print (att_actions.shape)
        #
        # for i in range(6):
        #     plt.plot(range(len(att_actions)), att_actions[:,i])
        #
        # plt.show()

        ### PLOT OBSERVATIONS

        # obs_buffer = np.array(env.env.inf.obs_buffer2)
        #
        # f, axarr = plt.subplots(2, 2)
        #
        # plot_pos = 0
        # plot_vel = 0
        # title = "robot 0 pos"
        # for i in range(24):
        #     if i == 6:
        #         plot_vel = 1
        #         title = "robot 0 vel"
        #     if i == 12:
        #         plot_vel = 0
        #         plot_pos = 1
        #         title = "robot 1 pos"
        #     if i == 18:
        #         plot_vel = 1
        #         title = "robot 1 vel"
        #
        #     axarr[plot_vel, plot_pos].plot(range(len(obs_buffer)), obs_buffer[:, i],
        #                                    label=int(i - (np.floor(i / 6) * 6)))
        #     axarr[plot_vel, plot_pos].set_title(title)
        #     axarr[plot_vel, plot_pos].legend()
        #
        # plt.show()


    def test_swordonly_mode():

        env = gym.make("ErgoFightStatic-Graphical-Fencing-Swordonly-v0")

        for k in range(3):
            observation = env.reset()
            print("init done")
            time.sleep(2)
            for i in range(40):
                if i % 10 == 0:
                    # action = env.action_space.sample() # this doesn't work
                    action = np.random.uniform(low=-1.0, high=1.0, size=(6))
                observation, reward, done, info = env.step(action)
                print(action, observation[0].shape, observation[1], reward, done)
                print(".")

        env.close()

        print('simulation ended. leaving in 3 seconds...')
        time.sleep(3)


    def test_swordonly_fat_mode():

        env = gym.make("ErgoFightStatic-Graphical-Fencing-Swordonly-Fat-NoMove-HalfRand-v0")

        for k in range(3):
            _ = env.reset()
            print("init done")
            time.sleep(2)
            for i in range(40):
                if i % 10 == 0:
                    # action = env.action_space.sample() # this doesn't work
                    action = np.random.uniform(low=-1.0, high=1.0, size=(6))
                observation, reward, done, info = env.step(action)

        env.close()

        print('simulation ended. leaving in 3 seconds...')
        time.sleep(3)


    def test_shield_nomove():
        env = gym.make("ErgoFightStatic-Graphical-Shield-NoMove-ThreequarterRand-v0")
        for k in range(3):
            _ = env.reset()
            print("init done")
            time.sleep(2)
            for i in range(40):
                if i % 10 == 0:
                    action = np.random.uniform(low=-1.0, high=1.0, size=(6))
                _ = env.step(action)

        env.close()

        print('simulation ended. leaving in 3 seconds...')
        time.sleep(3)


    def test_shield_move():
        env = gym.make("ErgoFightStatic-Headless-Shield-Move-ThreequarterRand-v0")
        obs_buf = []
        for _ in tqdm(range(10)):
            _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, rew, done, _ = env.step(action)
                obs_buf.append(obs)

        obs = np.array(obs_buf)
        print(obs.shape)

        print("robo 1 pos")
        print(obs[:, :6].max())
        print(obs[:, :6].min())
        print("robo 2 pos")
        print(obs[:, 12:18].max())
        print(obs[:, 12:18].min())

        print("robo 1 vel")
        print(obs[:, 6:12].max())
        print(obs[:, 6:12].min())
        print("robo 2 vel")
        print(obs[:, 18:24].max())
        print(obs[:, 18:25].min())

        env.close()


    def test_orientation():
        env = gym.make("ErgoFightStatic-Graphical-Shield-Move-HalfRand-v0")
        _ = env.reset()
        # for _ in tqdm(range(200)):
        #     action = [1,0,0,1,0,0]
        #     obs, rew, done, _ = env.step(action)

        for _ in tqdm(range(100)):
            action = [1, 1, -1, 1, 1, -1]
            obs, rew, done, _ = env.step(action)
        print(obs)

        for _ in tqdm(range(100)):
            action = [0, -1, 1, 0, 0, 0]
            obs, rew, done, _ = env.step(action)
        print(obs)


    test_orientation()

input = [1, 1, -1, 1, 1, -1]
output = [1, 1, -1, 1, 1, -1]

# robo 1 pos
# 0.8586047
# -0.5025676
# robo 2 pos
# 1.4171062
# -1.0011668
# robo 1 pos
# 46.274807
# -63.521507
# robo 2 pos
# 85.08411
# -97.77484
