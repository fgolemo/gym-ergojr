import gym
import numpy as np

class NormalizedObsWrapper(gym.ObservationWrapper):
    def _observation(self, observation):
        positions = np.sin(observation[:6])
        return np.hstack((positions, observation[6:]))


class NormalizedActWrapper(gym.ActionWrapper):
    def _reverse_action(self, action):
        act = []
        for i in range(6):
            # JOINT_LIMITS[i][0] should be negative for this to work
            dist = JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0]
            act.append(
                ((action[i] - JOINT_LIMITS[i][0]) / dist) * 2 - 1)  # *2-1 is so that it's in range [-1,1], not [0,1]
        return np.array(act)

    def _action(self, action):
        # we expect the actions to be in [-1,1] for -150 to +150 degrees (joint0) or -90 to +90 (joint1-5)
        # then we have to scale these inputs to be the actual degrees
        act = []
        for i in range(6):
            # JOINT_LIMITS[i][0] should be negative for this to work
            dist = JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0]
            act.append(((action[i] + 1) / 2) * dist + JOINT_LIMITS[i][0])
        return np.array(act)
