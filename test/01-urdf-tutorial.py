import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

p.setGravity(0, 0, -10)

planeId = p.loadURDF(URDF(get_scene("plane-big.urdf.xml")).get_path())

cubeStartPos = [0, 0, 0.5]  # RGB = xyz
cubeStartOrientation = p.getQuaternionFromEuler(
    [0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
# rotating a standing cylinder around the y axis, puts it flat onto the x axis

robot_file = URDF("urdfs/tutorial2.urdf", force_recompile=True).get_path()
robot = p.loadURDF(robot_file, cubeStartPos, cubeStartOrientation)

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

leftWheels = [6, 7]
rightWheels = [2, 3]

leftSlider = p.addUserDebugParameter("leftVelocity", -10, 10, 0)
rightSlider = p.addUserDebugParameter("rightVelocity", -10, 10, 0)
forceSlider = p.addUserDebugParameter("maxForce", -10, 10, 0)

for i in range(240 * 10):
    leftVelocity = p.readUserDebugParameter(leftSlider)
    rightVelocity = p.readUserDebugParameter(rightSlider)
    maxForce = p.readUserDebugParameter(forceSlider)

    for wheel in leftWheels:
        p.setJointMotorControl2(
            robot,
            wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=leftVelocity,
            force=maxForce)

    for wheel in rightWheels:
        p.setJointMotorControl2(
            robot,
            wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=rightVelocity,
            force=maxForce)

    p.stepSimulation()
    time.sleep(1. / 240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(robot)
print(cubePos, cubeOrn)
p.disconnect()
