import pybullet as p
import numpy as np

class DistanceBetweenObjects(object):
    def __init__(self, bodyA, bodyB, linkA, linkB):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.linkA = linkA
        self.linkB = linkB

    def query(self):
        contacts = p.getClosestPoints(bodyA=self.bodyA, bodyB=self.bodyB,
                                      linkIndexA=self.linkA, linkIndexB=self.linkB,
                                      distance=999)
        if len(contacts) == 0:
            return None

        return contacts[0][-2]

def sanitizeAction(action, alen):
    action = np.clip(action,-1,1)
    assert len(action) == alen
    return action
