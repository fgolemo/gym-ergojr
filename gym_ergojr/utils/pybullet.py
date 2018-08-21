import pybullet as p
import numpy as np


class DistanceBetweenObjects(object):
    def __init__(self, bodyA, bodyB, linkA, linkB):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.linkA = linkA
        self.linkB = linkB
        self.goal = (0, 0, 0)

    def query(self):
        for tries in range(5):
            posA = p.getLinkState(bodyUniqueId=self.bodyA, linkIndex=self.linkA)
            # rt = p.rayTest(self.goal,posA[1])
            # dist = np.linalg.norm(np.array(self.goal)-np.array(rt[3]))
            dist = np.linalg.norm(np.array(self.goal) - np.array(posA[0]))

            if np.isnan(dist):
                print(tries, "pos:", posA)
                print(tries, "goal:", self.goal)
                p.stepSimulation() # in case the sim just has a hiccup
                continue
            else:
                return dist

        # contacts = p.getClosestPoints(bodyA=self.bodyA, bodyB=self.bodyB,
        #                               linkIndexA=self.linkA, linkIndexB=self.linkB,
        #                               distance=999)
        # if len(contacts) == 0:
        #     return None
        # return contacts[0][-2]


def sanitizeAction(action, alen):
    action = np.clip(action, -1, 1)
    assert len(action) == alen
    return action
