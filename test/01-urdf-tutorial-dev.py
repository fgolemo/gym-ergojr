import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

# p.setGravity(0, 0, -10)

# planeId = p.loadURDF(URDF(get_scene("plane-big.urdf.xml")).get_path())

cubeStartPos = [0, 0, 0]  # RGB = xyz
cubeStartOrientation = p.getQuaternionFromEuler(
    [0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
# rotating a standing cylinder around the y axis, puts it flat onto the x axis

robot_file = URDF("urdfs/tutorial2.urdf.xml").get_path()
print("trying to load file:", robot_file)
robot = p.loadURDF(robot_file, cubeStartPos, cubeStartOrientation)

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

for i in range(240 * 10):

    p.stepSimulation()
    time.sleep(1. / 240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(robot)
print(cubePos, cubeOrn)
p.disconnect()
