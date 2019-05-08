import math
import pybullet as p
import time
import pybullet_data
import numpy as np

from gym_ergojr import get_scene
from gym_ergojr.sim.objects import Ball
from gym_ergojr.utils.math import RandomPointInHalfSphere
from gym_ergojr.utils.pybullet import DistanceBetweenObjects
from gym_ergojr.utils.urdf_helper import URDF

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(
    cameraDistance=0.75,
    cameraYaw=135,
    cameraPitch=-45,
    cameraTargetPosition=[0, 0, 0])

p.setGravity(0, 0, -10)
frequency = 100  # Hz
p.setTimeStep(1 / frequency)
p.setRealTimeSimulation(0)

planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 0]  # RGB = xyz
startOrientation = p.getQuaternionFromEuler(
    [0, 0, 0])  # rotated around which axis? # np.deg2rad(90)
# rotating a standing cylinder around the y axis, puts it flat onto the x axis

# params: object, radius, boxSize, cylinderLen, meshFile,
# meshScale, planeNormal, flags, rgbaColor, specularColor, visualFramePos,
# visualframeOrientation, pybClientID

ball = Ball()

xml_path = get_scene("ergojr-penholder")
#
robot_file = URDF(xml_path, force_recompile=True).get_path()
robot = p.loadURDF(robot_file, startPos, startOrientation, useFixedBase=1)
#
for i in range(p.getNumJoints(robot)):
  print(p.getJointInfo(robot, i))

motors = [3, 4, 6, 8, 10, 12]

debugParams = []

for i in range(len(motors)):
  motor = p.addUserDebugParameter("motor{}".format(i + 1), -1, 1, 0)
  debugParams.append(motor)

read_pos = p.addUserDebugParameter("read pos - slide right".format(i + 1), 0, 1,
                                   0)
read_pos_once = True

start = time.time()

rhis = RandomPointInHalfSphere(
    0.0, 0.0369, 0.0437, radius=0.2022, height=0.2610, min_dist=0.0477)

dist = DistanceBetweenObjects(bodyA=robot, bodyB=ball.id, linkA=13, linkB=-1)

for i in range(frequency * 30):
  motorPos = []
  for m in range(len(motors)):
    pos = (math.pi / 2) * p.readUserDebugParameter(debugParams[m])
    motorPos.append(pos)
    p.setJointMotorControl2(
        robot, motors[m], p.POSITION_CONTROL, targetPosition=pos)

  p.stepSimulation()
  time.sleep(1. / frequency)

  link_state = p.getLinkState(robot, 13)
  read_pos_val = p.readUserDebugParameter(read_pos)

  if read_pos_val >= 0.9 and read_pos_once:
    print(link_state[0])
    print(link_state[4])
    read_pos_once = False
  if read_pos_val <= 0.1:
    read_pos_once = True

  # if i % 30 == 0:
  #     new_pos = rhis.samplePoint()
  #     print (new_pos)
  #     p.changeConstraint(ball_cid, new_pos, maxForce=500)

  ball_dist = dist.query()
  print(ball_dist)

  if ball_dist <= 0.01:
    new_pos = rhis.samplePoint()
    ball.changePos(new_pos)
    for _ in range(10):
      p.stepSimulation()  # move ball

print(time.time() - start)

pos, orn = p.getBasePositionAndOrientation(robot)
print(pos, orn)
p.disconnect()
