import time
import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet
import pybullet_data
import matplotlib.pyplot as plt
from tqdm import trange

from gym_ergojr import get_scene
from gym_ergojr.utils.pybullet import getImg
from gym_ergojr.utils.urdf_helper import URDF

# simulator 0
p0 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p0.setAdditionalSearchPath(pybullet_data.getDataPath())

# simulator 1
p1 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p1.setAdditionalSearchPath(pybullet_data.getDataPath())

frequency = 100  # Hz

for p in [p0, p1]:
    p.setGravity(0, 0, -10)  # good enough
    p.setTimeStep(1 / frequency)
    p.setRealTimeSimulation(0)

# sim 0 / "p0" is our real environment
p0.loadURDF(URDF(get_scene("plane-alt")).get_path())

# sim 1 / "p1" is our simulated environment that we are trying to match
p1.loadURDF("plane.urdf")

# right now the robot arm is only loaded in the real env, mostly for scale. in the experiments we need to load it in both environments
xml_path = get_scene("ergojr-gripper")
robot_file = URDF(xml_path, force_recompile=True).get_path()
startOrientation = p0.getQuaternionFromEuler(
    [0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
robot = p0.loadURDF(robot_file, [0, 0, 0], startOrientation, useFixedBase=1)
motors = [3, 4, 6, 8, 10, 14]
for m in motors:
    p0.setJointMotorControl2(robot, m, p.POSITION_CONTROL, targetPosition=0)

scale = .05
shift = [0, 0.03, 0]
meshScale = np.array([1, 1, 1]) * scale

# the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
# the duckie shape is going to be the real object...
duckie_visual = p0.createVisualShape(
    shapeType=p0.GEOM_MESH,
    fileName="duck.obj",
    rgbaColor=[1, 1, 1, 1],
    specularColor=[0.4, .4, 0],
    visualFramePosition=shift,
    meshScale=meshScale)
duckie_collision = p0.createCollisionShape(
    shapeType=p0.GEOM_MESH,
    fileName="duck_vhacd.obj",
    collisionFramePosition=shift,
    meshScale=meshScale)

# and the cube is the simulated, matched object. Note how the colors here are going to be part of the parameters that we want to estimate
cube_visual = p1.createVisualShape(
    shapeType=p1.GEOM_BOX,
    halfExtents=meshScale,
    rgbaColor=[1, 0, 0, 1],
    specularColor=[0, 1, 0])
cube_collision = p1.createCollisionShape(
    shapeType=p1.GEOM_BOX, halfExtents=meshScale)

cam_view = p0.computeViewMatrix(
    cameraEyePosition=[.4, .3, .2],
    cameraTargetPosition=[0, .3, 0.1],
    cameraUpVector=[0, 0, 1])

cam_width = 200
cam_height = 200

cam_proj = p0.computeProjectionMatrixFOV(
    fov=90, aspect=cam_width / cam_height, nearVal=0.1, farVal=10)

# for the duckie this should only be done once
duckie = p0.createMultiBody(
    baseMass=.1,
    baseInertialFramePosition=[0, 0, 0],
    baseCollisionShapeIndex=duckie_visual,
    baseVisualShapeIndex=duckie_collision,
    basePosition=[0, 0.2, 0],
    baseOrientation=p0.getQuaternionFromEuler([0, 0, 0]))

# for the cube this next command is what actually puts the cube into the simulator.
# So this is the command that you need to run for every set of parameters that we wanna evaluate
cube = p1.createMultiBody(
    baseMass=.1,
    baseInertialFramePosition=[0, 0, 0],
    baseCollisionShapeIndex=cube_collision,
    baseVisualShapeIndex=cube_visual,
    basePosition=[0, 0.2, 0],
    baseOrientation=p1.getQuaternionFromEuler([0, 0, 0]))

# stabilize simulation
for i in range(20):
    p0.stepSimulation()
    p1.stepSimulation()

# live plotting
plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=2)
ax0 = ax[0].imshow(
    np.zeros((200, 200, 3)),
    interpolation='none',
    animated=True,
    label="real world")
ax1 = ax[1].imshow(
    np.zeros((200, 200, 3)), interpolation='none', animated=True, label="sim")
plot = plt.gca()

# here we're running the simulator for a fixed number of steps. However,
# for the experiments you would want to spawn the object, step, step, step, take a few snapshots, delete the object, repeat

steps = 1000
start = time.time()
for i in trange(steps):

    # step both simulators
    p0.stepSimulation()
    p1.stepSimulation()

    # get camera snapshot of real robot - in experiments this does NOT have to be run every cycle
    img0 = p0.getCameraImage(
        cam_width,
        cam_height,
        cam_view,
        cam_proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_NO_SEGMENTATION_MASK)

    # get camera snapshot of sim robot - this DOES have to be run for every sim step
    img1 = p1.getCameraImage(
        cam_width,
        cam_height,
        cam_view,
        cam_proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_NO_SEGMENTATION_MASK)

    ax0_img = getImg(img0, 200, 200)  # just placeholder for time estimation
    ax1_img = getImg(img1, 200, 200)
    ax0.set_data(ax0_img)
    ax1.set_data(ax1_img)
    fig.suptitle(f"step {i}", fontsize=16)
    # plot.plot([0])
    plt.pause(0.001)

    # time.sleep(1. / frequency)

    i += 1
    if i == 200:
        p0.applyExternalForce(
            objectUniqueId=duckie,
            linkIndex=-1,
            forceObj=[0, 15, 0],
            posObj=[0, 0, 0],
            flags=p0.WORLD_FRAME)
        p1.applyExternalForce(
            objectUniqueId=cube,
            linkIndex=-1,
            forceObj=[0, 15, 0],
            posObj=[0, 0, 0.05],
            flags=p0.WORLD_FRAME)

diff = (time.time() - start) / steps

print(f"{np.around(diff,3)}s/step, {np.around(1/diff,2)}Hz")

# experiment outline

# 1 take background pics from both envs -> background
# 1a add some noise to all real observations
# 1b average real obs

# 2 add object, take a few new real obs, average them
# 2a subtract background from new obs

# 3 create 10 random samples of [color, size, position], simulate them all out as single frames
# 3a optimize [c,s,p] via REINFORCE & EvoAlg until stable, compare (number of iterations necessary, goodness of fit)
# 3b calculate confidence of best solution (error)

# 4 apply force to real object, record frames
# 4a create 10 random samples of [c,s,p] in proximity of last estimate, apply same force to simulated object and record frames
# 4b optimize over all frames, minimize optical flow error with REINFORCE/EA, compare runtime and gof
# 4c calculate confidence of best solution (error)

# 5+ apply force in opposite direction, add new frames to dataset, optimize over all frames and repeat until stable
