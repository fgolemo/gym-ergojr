import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(cameraDistance=0.45,
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

xml_path = get_scene("ergojr-penholder-truebacklash.urdf.xml")

robot_file = URDF(xml_path, force_recompile=True).get_path()
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

# leftWheels = [6,7]
motors = [3, 5, 8]
# motors = [3, 5, 8, 11, 14, 17]

debugParams = []

for i in range(len(motors)):
    motor = p.addUserDebugParameter("motor{}".format(i + 1), -1, 1, 0)
    debugParams.append(motor)

forceSlider = p.addUserDebugParameter("maxForce",0,1000,10)

start = time.time()

for i in range(frequency * 30):
    motorPos = []
    for i in range(len(motors)):
        pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[i])
        motorPos.append(pos)
        p.setJointMotorControl2(robot, motors[i], p.VELOCITY_CONTROL, targetVelocity = pos,
        # p.setJointMotorControl2(robot, motors[i], p.POSITION_CONTROL, targetPosition=pos,
                                force = p.readUserDebugParameter(forceSlider))
    #
    #     targetVelocity = targetVel,
    #     force = maxForce

    p.stepSimulation()
    time.sleep(1. / frequency)

print(time.time() - start)

p.disconnect()
