import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
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


def load_puck():
    puck = p.loadURDF(
        robot_file,
        [np.random.uniform(-0.08, -0.14),
         np.random.uniform(0.05, 0.1), 0.0],
        startOrientation,
        useFixedBase=1)

    jointFrictionForce = .4
    for joint in [0, 1]:
        p.setJointMotorControl2(
            puck,
            joint,
            p.VELOCITY_CONTROL,
            force=jointFrictionForce,
            targetVelocity=0)

    dbo = DistanceBetweenObjects(puck, 1)

    return puck, dbo


puck, dbo = load_puck()


def load_goal():
    goal = np.array(
        [np.random.uniform(-.30, -.1),
         np.random.uniform(-.3, .1), .005])
    dbo.goal = goal

    target = p.createMultiBody(
        baseVisualShapeIndex=obj_visual, basePosition=goal)

    return target, goal


target, goal = load_goal()

for i in range(p.getNumJoints(puck)):
    print(p.getJointInfo(puck, i))

motors = [1, 3, 5]
motor_pos = [-.5, 1, .5]
debugParams = []

for i in range(len(motors)):
    motor = p.addUserDebugParameter("motor{}".format(i + 1), -1, 1,
                                    motor_pos[i])
    debugParams.append(motor)

start = time.time()

# Stepping frequency * 30 = run the simulation for 30 seconds
for i in range(frequency * 30):
    motorPos = []

    for j in range(len(motors)):
        pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[j])
        motorPos.append(pos)
        p.setJointMotorControl2(
            robot, motors[j], p.POSITION_CONTROL, targetPosition=pos)

    p.stepSimulation()

    dist = dbo.query()
    print(dist)

    time.sleep(1. / frequency)

    if i % 120 == 0 or dist < 0.01:
        p.removeBody(puck)
        puck, dbo = load_puck()

        p.removeBody(target)
        target, goal = load_goal()

print(time.time() - start)
p.disconnect()
