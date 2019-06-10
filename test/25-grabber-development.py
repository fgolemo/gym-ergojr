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
frequency = 20  # Hz
p.setTimeStep(1 / frequency)
p.setRealTimeSimulation(0)

robot_file = URDF(get_scene("plane"), force_recompile=True).get_path()
robot = p.loadURDF(robot_file)

startPos = [0, 0, 0]  # xyz
startOrientation = p.getQuaternionFromEuler(
    [0, 0, 0])  # rotated around which axis? # np.deg2rad(90)

xml_path = get_scene("ergojr-gripper")
robot_file = URDF(xml_path, force_recompile=True).get_path()
# Actually load the URDF file into simulation, make the base of the robot unmoving
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)

xml_path = get_scene("ergojr-gripper-cube")
robot_file = URDF(xml_path, force_recompile=True).get_path()

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))


def load_cube():
    # cube_pos = [0, 0.1, 0.015]
    # cube_pos = [0, 0.25, 0]
    cube_pos = [0, np.random.uniform(.1, .25), 0]
    cube_rot = p.getQuaternionFromEuler([
        0, 0, np.deg2rad(np.random.randint(0, 180))
    ])  # rotated around which axis? # np.deg2rad(90)

    cube = p.loadURDF(robot_file, cube_pos, cube_rot, useFixedBase=0)

    dbo = DistanceBetweenObjects(robot, 15, cube, 0)

    return cube, dbo


# def load_goal():
#   goal = np.array(
#       [np.random.uniform(-.30, -.1),
#        np.random.uniform(-.3, .1), .005])
#   dbo.goal = goal
#
#   target = p.createMultiBody(baseVisualShapeIndex=obj_visual, basePosition=goal)
#
#   return target, goal
#
#
# target, goal = load_goal()

cam_view = p.computeViewMatrix(
    cameraEyePosition=[.3, .05, .1],
    cameraTargetPosition=[0, .07, 0.05],
    cameraUpVector=[0, 0, 1])

cam_width = 400
cam_height = 300

cam_proj = p.computeProjectionMatrixFOV(
    fov=60, aspect=cam_width / cam_height, nearVal=0.1, farVal=0.8)

motors = [3, 4, 6, 8, 10, 14]
motor_pos = [0, 0, 0, 0, 0, 0]
debugParams = []

for i in range(len(motors)):
    motor = p.addUserDebugParameter("motor{}".format(i + 1), -1, 1,
                                    motor_pos[i])
    debugParams.append(motor)

cube, dbo = load_cube()

text = None

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

    img = p.getCameraImage(
        cam_width,
        cam_height,
        cam_view,
        cam_proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_NO_SEGMENTATION_MASK)

    dist = dbo.query()

    if text is not None:
        p.removeUserDebugItem(text)

    if dist > 0.007:
        dist_txt = f"distance: {dist}"
    else:
        dist_txt = f"YAY! {dist}"

    text = p.addUserDebugText(
        dist_txt, [0, 0, 0.02], textColorRGB=[1, 1, 0], textSize=2)

    time.sleep(1. / frequency)

    if i % 20 == 0 or dist < 0.007:
        p.removeBody(cube)
        cube, dbo = load_cube()

print(time.time() - start)
p.disconnect()
