import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
from gym_ergojr.utils.urdf_helper import URDF

MAX_VEL = 5  # not measured, but looks about right
MAX_FORCE = 1  # idk, seems to work


class AbstractRobot():

    def __init__(self, debug=False, frequency=100):
        self.debug = debug
        if debug:
            p.connect(p.GUI)  # or p.DIRECT for non-graphical version
            p.resetDebugVisualizerCamera(cameraDistance=0.7,
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

        self.robots =[]
        self.motor_ids = [3, 4, 6, 8, 10, 12] # this is consistent across different robots

        # self.addModel(robot_model1)

    def addModel(self, robot_model, pose=None):
        if pose is None:
            pose = [0, 0, 0, 0, 0, 0]
        startPos = pose[:3]  # RGB = xyz
        startOrientation = p.getQuaternionFromEuler(pose[3:])  # rotated around which axis? # np.deg2rad(90)
        # rotating a standing cylinder around the y axis, puts it flat onto the x axis

        xml_path = get_scene(robot_model)
        robot_file = URDF(xml_path, force_recompile=True).get_path()
        robot_id = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)
        self.robots.append(robot_id)

        if self.debug:
            print (robot_model)
            for i in range(p.getNumJoints(robot_id)):
                print(p.getJointInfo(robot_id, i))



    def act(self, actions, robot_id):
        actions_clipped = np.pi / 2 * np.clip(actions, -1, 1)
        p.setJointMotorControlArray(self.robots[robot_id], self.motor_ids,
                                    p.POSITION_CONTROL,
                                    targetPositions=actions_clipped,
                                    forces=[MAX_FORCE] * 6)

    def act2(self, actions, robot_id):
        actions_clipped = np.pi / 2 * np.clip(actions, -1, 1)
        for idx, act in enumerate(actions_clipped):
            p.setJointMotorControl2(self.robots[robot_id], self.motor_ids[idx],
                                    p.POSITION_CONTROL,
                                    targetPosition=act,
                                    force=MAX_FORCE,
                                    maxVelocity=MAX_VEL)

    def observe(self, robot_id):
        obs = p.getJointStates(self.robots[robot_id], self.motor_ids)
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

    def set(self, posvel, robot_id):
        assert len(posvel) == 12
        posvel_clipped = np.clip(posvel, -1, 1)
        posvel_clipped[:6] *= np.pi / 2
        for i in range(6):
            p.resetJointState(
                self.robots[robot_id],
                self.motor_ids[i],
                targetValue=posvel_clipped[i],
                targetVelocity=posvel_clipped[i + 6]
            )
