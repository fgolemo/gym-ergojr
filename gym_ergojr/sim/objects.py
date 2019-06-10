from gym_ergojr import get_scene
from gym_ergojr.utils.pybullet import DistanceBetweenObjects
from gym_ergojr.utils.urdf_helper import URDF
import pybullet as p
import numpy as np

PUSHER_GOAL_X = [-.2, -.1]
PUSHER_GOAL_Y = [-.1, .05]

PUSHER_PUCK_X = [-0.07, -0.10]
PUSHER_PUCK_Y = [0.05, 0.08]

GRIPPER_CUBE_Y = [.1, .25]

PUSHER_PUCK_X_NORM = [
    min(PUSHER_PUCK_X[0], PUSHER_PUCK_X[0]),
    max(PUSHER_PUCK_X[1], PUSHER_PUCK_X[1])
]
PUSHER_PUCK_Y_NORM = [
    min(PUSHER_PUCK_Y[0], PUSHER_PUCK_Y[0]),
    max(PUSHER_PUCK_Y[1], PUSHER_PUCK_Y[1])
]


class Ball(object):

    def __init__(self, scaling=0.02):
        self.scaling = scaling
        xml_path = get_scene("ball")
        self.ball_file = URDF(xml_path, force_recompile=False).get_path()
        self.hard_reset()

    def changePos(self, new_pos, speed=1):
        p.changeConstraint(
            self.ball_cid, new_pos, maxForce=25000 * self.scaling * speed)

    def hard_reset(self):
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.id = p.loadURDF(
            self.ball_file, [0, 0, .5],
            startOrientation,
            useFixedBase=0,
            globalScaling=self.scaling)
        self.ball_cid = p.createConstraint(
            self.id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0])
        p.changeConstraint(
            self.ball_cid, [0, 0, 0.1], maxForce=25000 * self.scaling)


class Puck(object):

    def __init__(self, friction=.4):
        self.friction = friction

        self.puck = None
        self.dbo = None
        self.target = None
        self.goal = None
        self.obj_visual = None

        xml_path = get_scene("ergojr-pusher-puck")
        self.robot_file = URDF(xml_path, force_recompile=True).get_path()

        # # GYM env has to do this
        # self.hard_reset()

    def add_puck(self):
        self.puck = p.loadURDF(
            self.robot_file, [
                np.random.uniform(PUSHER_PUCK_X[0], PUSHER_PUCK_X[1]),
                np.random.uniform(PUSHER_PUCK_Y[0], PUSHER_PUCK_Y[1]), 0.0
            ],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=1)

        for joint in [0, 1]:
            p.setJointMotorControl2(
                self.puck,
                joint,
                p.VELOCITY_CONTROL,
                force=self.friction,
                targetVelocity=0)

        self.dbo = DistanceBetweenObjects(self.puck, 1)

    def normalize_goal(self):
        x = (self.dbo.goal[0] - PUSHER_GOAL_X[0]) / (
            PUSHER_GOAL_X[1] - PUSHER_GOAL_X[0])
        y = (self.dbo.goal[1] - PUSHER_GOAL_Y[0]) / (
            PUSHER_GOAL_Y[1] - PUSHER_GOAL_Y[0])
        return np.array([x, y])

    def normalize_puck(self):
        pos = np.array(p.getLinkState(bodyUniqueId=self.puck,
                                      linkIndex=1)[0])[:2]
        x = (pos[0] - PUSHER_PUCK_X_NORM[0]) / (
            PUSHER_PUCK_X_NORM[1] - PUSHER_PUCK_X_NORM[0])
        y = (pos[1] - PUSHER_PUCK_Y_NORM[0]) / (
            PUSHER_PUCK_Y_NORM[1] - PUSHER_PUCK_Y_NORM[0])
        return np.array([x, y])

    def reset(self):
        if self.puck is not None:
            p.removeBody(self.puck)

        self.add_puck()

        if self.target is not None:
            p.removeBody(self.target)

        self.add_target()

    def hard_reset(self):
        self.add_puck()
        self.obj_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.02, length=0.01, rgbaColor=[0, 1, 0, 1])
        self.add_target()

    def add_target(self):
        self.dbo.goal = np.array([
            np.random.uniform(PUSHER_GOAL_X[0], PUSHER_GOAL_X[1]),
            np.random.uniform(PUSHER_GOAL_Y[0], PUSHER_GOAL_Y[1]), 0
        ])

        self.target = p.createMultiBody(
            baseVisualShapeIndex=self.obj_visual, basePosition=self.dbo.goal)


class Cube(object):

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.cube = None
        self.dbo = None

        xml_path = get_scene("ergojr-gripper-cube")
        self.robot_file = URDF(xml_path, force_recompile=True).get_path()

        # # GYM env has to do this
        # self.hard_reset()

    def add_cube(self, y=None):
        if y is None:
            y = np.random.uniform(.1, .25)

        cube_pos = [0, y, 0]
        cube_rot = p.getQuaternionFromEuler([
            0, 0, np.deg2rad(np.random.randint(0, 180))
        ])  # rotated around which axis? # np.deg2rad(90)

        self.cube = p.loadURDF(
            self.robot_file, cube_pos, cube_rot, useFixedBase=1)

        self.dbo = DistanceBetweenObjects(self.robot_id, 15, self.cube, 0)

    def normalize_cube(self):
        _, posB = self.dbo.query(True)
        x = posB[0]
        y = (posB[1] - GRIPPER_CUBE_Y[0]) / (
            GRIPPER_CUBE_Y[1] - GRIPPER_CUBE_Y[0])
        z = posB[2]
        return np.array([x, y, z])

    def reset(self):
        if self.cube is not None:
            p.removeBody(self.cube)

        self.add_cube()
