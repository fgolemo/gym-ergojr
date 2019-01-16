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

# planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 0]  # RGB = xyz
startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
# rotating a standing cylinder around the y axis, puts it flat onto the x axis

robot_file = URDF("urdfs/backlash-test2c.xacro.xml", force_recompile=True).get_path()
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

# motors = [0,5]
# motors = [0,3]
motors = [0,2]

debugParams = []
#
for i in range(len(motors)):
    motor = p.addUserDebugParameter("motor{}".format(i + 1), -1, 1, 0)
    debugParams.append(motor)

# forceSlider = p.addUserDebugParameter("maxForce",-10,10,0)

start = time.time()
#
for i in range(frequency * 60):
    motorPos = []
    for i in range(len(motors)):
        pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[i])
        motorPos.append(pos)
        p.setJointMotorControl2(robot, motors[i], p.POSITION_CONTROL, targetPosition=pos)

    # maxForce = p.readUserDebugParameter(forceSlider)

    p.stepSimulation()
    time.sleep(1. / frequency)

print(time.time() - start)

p.disconnect()


# BL-vanilla,    motor -> (revolute/cont) -> wheel -> (revolute) -> wheel - > (fixed) -> arm
# Ergo-vanilla,  motor -> (revolute/cont) -> arm
# EBL-vanilla,   motor -> (revolute/cont) -> dummy -> (revolute) -> arm
