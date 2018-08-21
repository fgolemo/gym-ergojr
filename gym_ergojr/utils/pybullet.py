import pybullet as p
import numpy as np

class DistanceBetweenObjects(object):
    def __init__(self, bodyA, bodyB, linkA, linkB):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.linkA = linkA
        self.linkB = linkB
        self.goal = (0,0,0)

    def query(self):
        posA = p.getLinkState(bodyUniqueId=self.bodyA, linkIndex=self.linkA)
        _ = [float(x) for x in posA[0]]
        # rt = p.rayTest(self.goal,posA[1])
        # dist = np.linalg.norm(np.array(self.goal)-np.array(rt[3]))
        dist = np.linalg.norm(np.array(self.goal)-np.array(posA[0]))
        if np.isnan(dist):
            print ("pos:",posA)
            print ("goal:",self.goal)
            raise Exception("NaN dist")

        return dist


        # contacts = p.getClosestPoints(bodyA=self.bodyA, bodyB=self.bodyB,
        #                               linkIndexA=self.linkA, linkIndexB=self.linkB,
        #                               distance=999)
        # if len(contacts) == 0:
        #     return None
        # return contacts[0][-2]


def sanitizeAction(action, alen):
    action = np.clip(action,-1,1)
    assert len(action) == alen
    return action
