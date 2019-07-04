import math
import os

import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
from gym_ergojr.models import MODEL_PATH
from gym_ergojr.utils.pybullet import DistanceBetweenObjects
from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(
    cameraDistance=0.4,
    cameraYaw=135,
    cameraPitch=-35,
    cameraTargetPosition=[0, 0.05, 0])

p.setGravity(0, 0, -10)  # good enough
frequency = 100  # Hz
p.setTimeStep(1 / frequency)
p.setRealTimeSimulation(0)

planeId = p.loadURDF(URDF(get_scene("plane-big.urdf.xml")).get_path())

startPos = [0, 0, 0]  # xyz
startOrientation = p.getQuaternionFromEuler(
    [0, 0, 0])  # rotated around which axis? # np.deg2rad(90)

xml_path = get_scene("ergojr-pusher")
robot_file = URDF(xml_path, force_recompile=True).get_path()

# Actually load the URDF file into simulation, make the base of the robot unmoving
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

xml_path = get_scene("ergojr-pusher-puck")
robot_file = URDF(xml_path, force_recompile=True).get_path()

# Actually load the URDF file into simulation, make the base of the robot unmoving

obj_visual = p.createVisualShape(
    p.GEOM_CYLINDER, radius=0.02, length=0.01, rgbaColor=[0, 1, 0, 1])

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

motors = [1, 3, 5]

measurements = []
actions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]) * (math.pi / 2)

# Stepping frequency * 30 = run the simulation for 30 seconds
for i in range(3):
    motorPos = []

    for _ in range(30):

        for m_idx in range(3):
            p.setJointMotorControl2(
                robot,
                motors[m_idx],
                p.POSITION_CONTROL,
                targetPosition=actions[len(measurements)][m_idx])

        p.stepSimulation()
        time.sleep(0.001)

    p.stepSimulation()
    measurements.append(p.getLinkState(robot, 6)[0])
    time.sleep(1)

p.disconnect()

measurements = np.array(measurements)[:, :2]
np.savez(os.path.join(MODEL_PATH, "calibration-pusher-sim.npz"), measurements)
