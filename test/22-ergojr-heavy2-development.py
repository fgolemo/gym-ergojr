import math
import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from gym_ergojr import get_scene
from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(cameraDistance=4,
                             cameraYaw=135,
                             cameraPitch=-45,
                             cameraTargetPosition=[0, 0, 0])

p.setGravity(0, 0, -10)
frequency = 100  # Hz
p.setTimeStep(1 / frequency)
p.setRealTimeSimulation(0)

plane = URDF(get_scene("plane-big.urdf.xml")).get_path()
planeId = p.loadURDF(plane)

startPos = [0, 0, 0]  # RGB = xyz
startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
# rotating a standing cylinder around the y axis, puts it flat onto the x axis

xml_path = get_scene("ergojr-penholder-heavy2")

robot_file = URDF(xml_path, force_recompile=True).get_path()
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))
#
# c = p.createConstraint(robot,6,robot,7,jointType=p.JOINT_GEAR,jointAxis =[1,0,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
# p.changeConstraint(c, gearRatio=.5, maxForce=100000)

# leftWheels = [6,7]
# motors = [3, 5, 8]
# motors = [3, 5, 8, 11, 14, 17]
motors = [3, 6, 9, 12, 15, 18]

backlash = [4, 5, 8, 11, 14, 17]

runtime = frequency * 10

debugParams = []

for i in range(len(motors)):
    motor = p.addUserDebugParameter("motor{}".format(i + 1), -1, 1, 0)
    debugParams.append(motor)

# forceSlider = p.addUserDebugParameter("maxForce",0,1000,10)

start = time.time()

# for b in backlash:
#         p.setJointMotorControl2(robot, b, p.POSITION_CONTROL, targetPosition=0, force=1000)

joint_tracking = np.zeros((runtime, 7), np.float32)
joint_to_track = 2

for i in range(runtime):
    motorPos = []
    for j in range(len(motors)):
        pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[j])
        motorPos.append(pos)
        # p.setJointMotorControl2(robot, motors[i], p.VELOCITY_CONTROL, targetVelocity = pos)
        p.setJointMotorControl2(robot, motors[j], p.POSITION_CONTROL, targetPosition=pos, force=1000)
        #                         force = p.readUserDebugParameter(forceSlider))
    pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[joint_to_track-1])
    jpos, jvel, _, jtor = p.getJointState(robot, motors[joint_to_track-1])
    bpos, bvel, _, btor = p.getJointState(robot, backlash[joint_to_track-1])
    joint_tracking[i, :] = [pos, jpos, bpos, jvel, bvel, jtor, btor]

    p.stepSimulation()
    time.sleep(1. / frequency)


print(time.time() - start)

p.disconnect()

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(np.arange(0,runtime), joint_tracking[:,0], label="target pos")
axarr[0].plot(np.arange(0,runtime), joint_tracking[:,1], label="actual pos")
axarr[0].plot(np.arange(0,runtime), joint_tracking[:,2], label="backlash pos")
axarr[0].legend()
axarr[1].plot(np.arange(0,runtime), joint_tracking[:,3], label="actual vel")
axarr[1].plot(np.arange(0,runtime), joint_tracking[:,4], label="backlash vel")
axarr[1].legend()
axarr[2].plot(np.arange(0,runtime), joint_tracking[:,5], label="actual tor")
axarr[2].plot(np.arange(0,runtime), joint_tracking[:,6], label="backlash tor")
axarr[2].legend()
plt.tight_layout()
plt.show()
