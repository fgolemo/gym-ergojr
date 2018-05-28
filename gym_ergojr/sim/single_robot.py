import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
from gym_ergojr.utils.urdf_helper import URDF

MAX_VEL = 5  # not measured, but looks about right
MAX_FORCE = 1  # idk, seems to work


class SingleRobot():

    def __init__(self, robot_model="ergojr-sword", debug=False, frequency=100):
        if debug:
            p.connect(p.GUI)  # or p.DIRECT for non-graphical version
            p.resetDebugVisualizerCamera(cameraDistance=0.4,
                                         cameraYaw=135,
                                         cameraPitch=-45,
                                         cameraTargetPosition=[0, 0, 0])
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional for ground
        p.setGravity(0, 0, -10)
        p.setTimeStep(1 / frequency)
        p.setRealTimeSimulation(0)

        p.loadURDF("plane.urdf")

        startPos = [0, 0, 0]  # RGB = xyz
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
        # rotating a standing cylinder around the y axis, puts it flat onto the x axis

        xml_path = get_scene(robot_model)
        robot_file = URDF(xml_path, force_recompile=True).get_path()
        self.robot_id = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

        if debug:
            for i in range(p.getNumJoints(self.robot_id)):
                print(p.getJointInfo(self.robot_id, i))

        self.motor_ids = [3, 4, 6, 8, 10, 12]

    def act(self, actions):
        actions_clipped = np.pi / 2 * np.clip(actions, -1, 1)
        p.setJointMotorControlArray(self.robot_id, self.motor_ids,
                                    p.POSITION_CONTROL,
                                    targetPositions=actions_clipped,
                                    forces=[MAX_FORCE] * 6)

    def act2(self, actions):
        actions_clipped = np.pi / 2 * np.clip(actions, -1, 1)
        for idx, act in enumerate(actions_clipped):
            p.setJointMotorControl2(self.robot_id, self.motor_ids[idx],
                                    p.POSITION_CONTROL,
                                    targetPosition=act,
                                    force=MAX_FORCE,
                                    maxVelocity=MAX_VEL)

    def observe(self):
        obs = p.getJointStates(self.robot_id, self.motor_ids)
        pos = [x[0] for x in obs]
        vel = [x[1] for x in obs]
        return self.normalize(np.array(pos + vel))

    def normalize(self, posvel):
        assert len(posvel) == 12
        pos_norm = (posvel[:6] + np.pi / 2) / np.pi
        vel_norm = (posvel[6:] + MAX_VEL) / (MAX_VEL * 2)
        posvel_norm = np.hstack((pos_norm, vel_norm))
        posvel_shifted = posvel_norm * 2 - 1

        return posvel_shifted

    def close(self):
        p.disconnect()

    def step(self):
        p.stepSimulation()

    def set(self, posvel):
        assert len(posvel) == 12
        posvel_clipped = np.clip(posvel, -1, 1)
        posvel_clipped[:6] *= np.pi / 2
        for i in range(6):
            p.resetJointState(
                self.robot_id,
                self.motor_ids[i],
                targetValue=posvel_clipped[i],
                targetVelocity=posvel_clipped[i + 6]
            )
