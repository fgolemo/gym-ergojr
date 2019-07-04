import pybullet as p
import numpy as np


class DistanceBetweenObjects(object):

    def __init__(self, bodyA, linkA, bodyB=None, linkB=None):
        self.bodyA = bodyA
        self.linkA = linkA

        self.bodyB = bodyB
        self.linkB = linkB

        self.goal = None

    def query(self, return_posB=False):
        assert self.goal is not None or (self.bodyB is not None and
                                         self.linkB is not None)

        for tries in range(5):
            posA = p.getLinkState(bodyUniqueId=self.bodyA, linkIndex=self.linkA)
            # rt = p.rayTest(self.goal,posA[1])
            # dist = np.linalg.norm(np.array(self.goal)-np.array(rt[3]))
            if self.goal is not None:
                dist = np.linalg.norm(
                    np.array(self.goal)[:2] - np.array(posA[0])[:2])
            else:
                posB = p.getLinkState(
                    bodyUniqueId=self.bodyB, linkIndex=self.linkB)
                dist = np.linalg.norm(np.array(posB[0]) - np.array(posA[0]))

            if np.isnan(dist):
                print(tries, "pos:", posA)
                print("body {}, link {}".format(self.bodyA, self.linkA))
                p.stepSimulation()
                continue
            else:
                if not return_posB:
                    return dist
                else:
                    return dist, posB[0]

        # contacts = p.getClosestPoints(bodyA=self.bodyA, bodyB=self.bodyB,
        #                               linkIndexA=self.linkA, linkIndexB=self.linkB,
        #                               distance=999)
        # if len(contacts) == 0:
        #     return None
        # return contacts[0][-2]


class Cam(object):

    def __init__(self,
                 width=400,
                 height=300,
                 fov=60,
                 pos=[.5, .5, .5],
                 look_at=[0, 0, 0]):
        self.width = width
        self.heigt = height

        self.view = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=look_at,
            cameraUpVector=[0, 0, 1])

        self.proj = p.computeProjectionMatrixFOV(
            fov=fov, aspect=width / height, nearVal=0.1, farVal=0.8)

        # if this should ever trigger, the official PyPi version (pip install pybullet) is compiled with Numpy headers
        assert p.isNumpyEnabled(
        )  # because if not, copying the pixels of the camera image from C++ to python takes forever

    def snap(self):
        img = p.getCameraImage(
            self.width,
            self.heigt,
            self.view,
            self.proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK)

        return getImg(img, self.heigt, self.width)


def getImg(img, height, width):
    rgb = img[2]
    rgb = np.reshape(rgb, (height, width, 4))
    rgb = rgb * (1. / 255.)
    rgb = rgb[:, :, :3]
    return rgb


def sanitizeAction(action, alen):
    action = np.clip(action, -1, 1)
    assert len(action) == alen
    return action
