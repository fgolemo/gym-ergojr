import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(cameraDistance=0.3,
                             cameraYaw=135,
                             cameraPitch=-45,
                             cameraTargetPosition=[0,0,0])

p.setGravity(0, 0, -10)
frequency = 100 # Hz
p.setTimeStep(1/frequency)
p.setRealTimeSimulation(0)


planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 0]  # RGB = xyz
startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
# rotating a standing cylinder around the y axis, puts it flat onto the x axis

robot_file = URDF("urdfs/ergojr", force_recompile=True).get_path()
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

# leftWheels = [6,7]
motors = [2,3]

motor1 = p.addUserDebugParameter("motor1",-1,1,0)
motor2 = p.addUserDebugParameter("motor2",-1,1,0)
# forceSlider = p.addUserDebugParameter("maxForce",-10,10,0)

start = time.time()

for i in range(frequency*10):
    motor1pos = (math.pi/2) * p.readUserDebugParameter(motor1)
    motor2pos = (math.pi/2) * p.readUserDebugParameter(motor2)
    # maxForce = p.readUserDebugParameter(forceSlider)

    # for wheel in leftWheels:
    #     p.setJointMotorControl2(robot, wheel, p.VELOCITY_CONTROL, targetVelocity=leftVelocity, force=maxForce)
    #
    p.setJointMotorControl2(robot, motors[0], p.POSITION_CONTROL, targetPosition=motor1pos)
    p.setJointMotorControl2(robot, motors[1], p.POSITION_CONTROL, targetPosition=motor2pos)

    p.stepSimulation()
    time.sleep(1. / frequency )

print (time.time() - start)

pos, orn = p.getBasePositionAndOrientation(robot)
print(pos, orn)
p.disconnect()
