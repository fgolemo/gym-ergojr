import numpy as np


class RandomPointInHalfSphere(object):
    def __init__(self, center_x, center_y, center_z, radius, height=None, min_dist=None, halfsphere=False):
        self.center = np.array([center_x, center_y, center_z])
        self.radius = radius
        self.halfsphere = halfsphere

        self.height = radius
        if height is not None:
            self.height = height

        self.min_dist = 0
        if min_dist is not None:
            self.min_dist = min_dist

    def samplePoint(self):
        while True:
            point_x = np.random.uniform(low=-self.radius, high=self.radius, size=1)
            low = 0
            if self.halfsphere:
                low = -self.radius * 2 / 3  # (because the robot can't go back as far as forward)
            point_y = np.random.uniform(low=low, high=self.radius, size=1)
            point_z = np.random.uniform(low=0, high=self.height, size=1)
            point = np.hstack((point_x, point_y, point_z))

            dist = np.linalg.norm(point)

            if dist > self.radius or dist < self.min_dist:
                continue

            point += self.center

            return point

    def sampleSimplePoint(self):
        while True:
            low = 0
            if self.halfsphere:
                low = -self.radius * 2 / 3  # (because the robot can't go back as far as forward)
            point_y = np.random.uniform(low=low, high=self.radius, size=1)
            point_z = np.random.uniform(low=0, high=self.height, size=1)
            point = np.hstack(([0], point_y, point_z))

            dist = np.linalg.norm(point)

            if dist > self.radius or dist < self.min_dist:
                continue

            point += self.center

            return point

    def normalize(self, point):
        point_ = point.copy()
        point_[:2] = (point_[:2] - self.center[:2] + self.radius) / (2 * self.radius)  # now it's in [0,1]
        point_[:2] = point_[:2] * 2 - 1

        point_[2] = (point_[2] - self.center[2]) / (self.height)  # now it's in [0,1]
        point_[2] = point_[2] * 2 - 1

        return point_
