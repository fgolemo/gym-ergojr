import numpy as np


class RandomPointInHalfSphere(object):
    def __init__(self, center_x, center_y, center_z, radius, height=None, min_dist=None):
        self.center = np.array([center_x, center_y, center_z])
        self.radius = radius

        self.height = radius
        if height is not None:
            self.height = height

        self.min_dist = 0
        if min_dist is not None:
            self.min_dist = min_dist


    def samplePoint(self):
        while True:
            point_xy = np.random.uniform(low=-self.radius, high=self.radius, size=2)
            point_z = np.random.uniform(low=0, high=self.height, size=1)
            point = np.hstack((point_xy, point_z))

            dist = np.linalg.norm(point)

            if dist > self.radius or dist < self.min_dist:
                continue

            point += self.center

            return point

