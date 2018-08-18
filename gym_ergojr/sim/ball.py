from gym_ergojr import get_scene
from gym_ergojr.utils.urdf_helper import URDF
import pybullet as p


class Ball(object):
    def __init__(self):
        xml_path = get_scene("ball")
        self.ball_file = URDF(xml_path, force_recompile=False).get_path()
        self.hard_reset()

    def changePos(self, new_pos):
        p.changeConstraint(self.ball_cid, new_pos, maxForce=500)

    def hard_reset(self):
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.id = p.loadURDF(self.ball_file, [0, 0, .5], startOrientation, useFixedBase=0, globalScaling=0.02)
        self.ball_cid = p.createConstraint(self.id, -1, -1, -1, p.JOINT_FIXED,
                                           jointAxis=[0, 0, 0],
                                           parentFramePosition=[0, 0, 0],
                                           childFramePosition=[0, 0, 0])
        p.changeConstraint(self.ball_cid, [0, 0, 0.1], maxForce=500)

