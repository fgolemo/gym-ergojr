import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(cameraDistance=0.7,
                             cameraYaw=135,
                             cameraPitch=-45,
                             cameraTargetPosition=[0, 0, 0])

p.setGravity(0, 0, -10)
frequency = 100  # Hz
p.setTimeStep(1 / frequency)
p.setRealTimeSimulation(0)

planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 0]  # RGB = xyz
startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
# rotating a standing cylinder around the y axis, puts it flat onto the x axis

xml_path = get_scene("ergojr-sword")

robot_file = URDF(xml_path, force_recompile=True).get_path()
robot1 = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

startPos = [0, 0.37, 0]  # RGB = xyz
startOrientation = p.getQuaternionFromEuler([0, 0, np.pi])

xml_path = get_scene("ergojr-shield")

robot_file = URDF(xml_path, force_recompile=True).get_path()
robot2 = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

robots = [robot1, robot2]

for robot in robots:
    for i in range(p.getNumJoints(robot)):
        print(p.getJointInfo(robot, i))

motors = [3, 4, 6, 8, 10, 12]
debugParams = []

for r in range(1,3):
    for i in range(len(motors)):
        motor = p.addUserDebugParameter("robot{} motor{}".format(r,i + 1), -1, 1, 0)
        debugParams.append(motor)

# forceSlider = p.addUserDebugParameter("maxForce",-10,10,0)

start = time.time()

for i in range(frequency * 30):
    motorPos = []
    for r in range(1,3):
        for i in range(len(motors)):
            pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[i+(len(motors)*(r-1))])
            motorPos.append(pos)
            p.setJointMotorControl2(robots[r-1], motors[i], p.POSITION_CONTROL, targetPosition=pos)

    # maxForce = p.readUserDebugParameter(forceSlider)
    hits = p.getContactPoints(robots[0], robots[1], 14, 14)
    if len(hits) > 0:
        print ("hit","."*np.random.randint(1,10))


    p.stepSimulation()
    time.sleep(1. / frequency)

print(time.time() - start)

pos, orn = p.getBasePositionAndOrientation(robot1)
print(pos, orn)
p.disconnect()
