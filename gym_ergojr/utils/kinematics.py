import math

import numpy as np

""" All credit for this code goes to Maxime Chevalier-Boisvert, github @maximecb
"""


def gen_rot_matrix(axis, angle):
    """
    Rotation matrix for a counterclockwise rotation around a given axis
    The angle should be in radians
    """

    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)

    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
    ])


def forward_kinematics(angles):
    """
    Compute the XYZ position of the end of the gripper relative to the base.
    """

    # real robot

    # L0 = 0.033 # Height of the base to first segment
    # L1 = 0.025 # Segment 1
    # L2 = 0.055 # Length of segment 2
    # L3 = 0.035 # Length of segment 3
    # L4 = 0.050 # Length of segment 4
    # # L5 = 0.130 # Segment 5 + gripper length
    # L5 = 0.055 # Segment 5
    # L6 = 0.042 # Segment 6 + center of tip

    # sim

    # Height of the base to first segment
    L0_y = .038  # base offset
    L0_z = .005 + .003 + .027  # base + base2 + motor1

    # Segment 1, upward
    L1 = 0.024

    # Length of segment 2, upward
    L2 = .019 + .035  # plastic + motor

    # Length of segment 3, upward
    L3 = .026 + .002 + .015 / 2

    # Length of segment 4, forward
    L4 = .013 + .035

    # Segment 5, forward
    L5 = .019 + .035

    # Segment 6 + center of tip
    L6 = .042  # technically correct but slightly too long

    # Segment 6 and center of tip
    pos = np.array((0, L6, 0))
    m = gen_rot_matrix(np.array((1, 0, 0)), angles[5] * math.pi / 180)
    pos = m.dot(pos)

    # Segment 5 and gripper plates
    pos += np.array((0, L5, 0))
    m = gen_rot_matrix(np.array((1, 0, 0)), angles[4] * math.pi / 180)
    pos = m.dot(pos)

    # Segment 4, forward segment
    # Segment 3, upwards
    pos += np.array((0, L4, 0))
    pos += np.array((0, 0, L3))
    m = gen_rot_matrix(np.array((1, 0, 0)), angles[2] * math.pi / 180)
    pos = m.dot(pos)

    # Segment 2, upwards
    pos += np.array((0, 0, L2))
    m = gen_rot_matrix(np.array((1, 0, 0)), angles[1] * math.pi / 180)
    pos = m.dot(pos)

    # Segment 1, upwards, rotation around Y
    pos += np.array((0, 0, L1))
    # m = gen_rot_matrix(np.array((0, 1, 0)), -angles[0] * math.pi/180)
    m = gen_rot_matrix(np.array((0, 0, 1)), 0)  # fixed to zero because joint locked
    pos = m.dot(pos)

    # Base height
    pos += np.array((0, L0_y, L0_z))

    return pos


def inverse_kinematics(target, samples=500):
    """
    Find the joint angles best matching a target gripper position
    """

    best_dist = math.inf
    best_angles = [0] * 6

    for i in range(samples):
        angles = np.random.uniform(-90, 90, 6)
        angles[[0, 3]] = 0
        pos = forward_kinematics(angles)
        dist = np.linalg.norm(pos - target)
        # print (pos, dist)

        if dist < best_dist:
            best_dist = dist
            best_angles = angles

    return np.array(best_angles)
