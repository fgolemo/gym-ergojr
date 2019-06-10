import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
from gym_ergojr.utils.urdf_helper import URDF

# Create the bullet physics engine environment
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(
    cameraDistance=0.45,
    cameraYaw=135,
    cameraPitch=-45,
    cameraTargetPosition=[0, 0, 0])
p.setGravity(0, 0, -10)  # good enough
frequency = 100  # Hz
p.setTimeStep(1 / frequency)
p.setRealTimeSimulation(0)

# This loads the checkerboard background
planeId = p.loadURDF(URDF(get_scene("plane-big.urdf.xml")).get_path())

# Robot model starting position
startPos = [0, 0, 0]  # xyz
startOrientation = p.getQuaternionFromEuler(
    [0, 0, 0])  # rotated around which axis? # np.deg2rad(90)

# The robot needs to be defined in a URDF model file. But the model file is clunky, so there is a macro language, "XACRO",
# that can be compiled to URDF but that's easier to read and that contains if/then/else statements, variables, etc.
# This next part looks for XACRO files with the given name and then force-recompiles it to URDF
# ("force" because if there is already a URDF file with the right name, it usually doesn't recompile - this is just for development)
xml_path = get_scene("ergojr-sword")
robot_file = URDF(xml_path, force_recompile=True).get_path()

# Actually load the URDF file into simulation, make the base of the robot unmoving
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

# Query all joints and print their infos. A robot URDF file consists of joints (moving parts) and links (visible parts).
# If you have a robot arm, it's usually base (link) -> joint1 -> link1 -> joint2 -> link2 -> joint3 -> link3 -> etc.
# But if you have a car for example, it can be
#   base (link) -> joint -> chassis (link)
#                           -> joint_wheel_left -> wheel_left (link)
#                           -> joint_wheel_right -> wheel_right (link)
# (i.e. it's an acyclical graph; a link can have many joints attached)
for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

# Indices of all the motors we can actually move. This information comes from the print statemtnt above.
motors = [3, 4, 6, 8, 10, 12]

# Container for debug inputs
debugParams = []

# In the user interface, create a slider for each motor to control them separately.
for i in range(len(motors)):
    motor = p.addUserDebugParameter("motor{}".format(i + 1), -1, 1, 0)
    debugParams.append(motor)

start = time.time()

# Stepping frequency * 30 = run the simulation for 30 seconds
for i in range(frequency * 30):
    motorPos = []

    # At each timestep, read the values from the motor sliders and set the motors to that position
    # via position control. The motors have an internal PID controller, that will take over and calculate
    # the force and direction that is necessary to reach the new target position from the current position it's in.
    # In practice, this doesn't need to be called so often. As soon as you send a new motor position command to the
    # motor via `setJointMotorControl2`, it will go there over time. No need to send the same command twice.
    # The only reason it's here in the loop, is so that user can slide the motor values around and the robot moves in
    # realtime.
    for i in range(len(motors)):
        pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[i])
        motorPos.append(pos)
        p.setJointMotorControl2(
            robot, motors[i], p.POSITION_CONTROL, targetPosition=pos)

    p.stepSimulation()

    # If you don't sleep, the simulation will run much faster, up to several thousand cycles per second.
    time.sleep(1. / frequency)

print(time.time() - start)

# Close the Bullet physics server
p.disconnect()
