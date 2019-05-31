import math
import pybullet as p
import time
import pybullet_data
import numpy as np
import os.path as osp
from gym_ergojr import get_scene
from gym_ergojr.utils.libstfu import stdout_redirected, stdout_noop
from gym_ergojr.utils.urdf_helper import URDF
from xml.etree import ElementTree as et

NORM_VEL = {"default": 18, "heavy": 38}
MAX_VEL = {"default": 18, "heavy": 1000}  # not measured, but looks about right
MAX_FORCE = {"default": 1, "heavy": 1000}
MOTOR_DIRECTIONS_DEFAULT = [1, -1, -1, 1, -1,
                            -1]  # how do the motors turn on real robot
MOTOR_DIRECTIONS_PUSHER = [1, 1, 1]  # how do the motors turn on real pusher
NAMESPACE = {'xacro': 'http://www.ros.org/wiki/xacro'}  # add more as needed
et.register_namespace("xacro", NAMESPACE["xacro"])


class AbstractRobot():

    def __init__(self,
                 debug=False,
                 frequency=100,
                 backlash=None,
                 heavy=False,
                 new_backlash=None,
                 silent=False,
                 gripper=False):
        self.debug = debug
        self.frequency = frequency
        self.backlash = backlash
        self.heavy = heavy
        self.gripper = gripper
        self.new_backlash = new_backlash
        self.output_handler = stdout_noop
        if silent:
            self.output_handler = stdout_redirected

        with self.output_handler():
            if debug:
                p.connect(p.GUI)  # or p.DIRECT for non-graphical faster version
                dist = .7
                if self.heavy:
                    dist = 50
                p.resetDebugVisualizerCamera(
                    cameraDistance=dist,
                    cameraYaw=135,
                    cameraPitch=-45,
                    cameraTargetPosition=[0, 0, 0])
            else:
                p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # optional for ground

        self.robots = []
        if not self.heavy:
            self.motor_ids = [
                3, 4, 6, 8, 10, 12
            ]  # this is consistent across different normal robots
        if self.heavy:
            self.motor_ids = [
                3, 6, 9, 12, 15, 18
            ]  # this is consistent across different heavy robots
        if self.gripper:
            self.motor_ids = [3, 4, 6, 8, 10, 14]
        self.debug_text = None

    def addModel(self, robot_model, pose=None):
        if pose is None:
            pose = [0, 0, 0, 0, 0, 0]
        startPos = pose[:3]  # RGB = xyz
        startOrientation = p.getQuaternionFromEuler(
            pose[3:])  # rotated around which axis? # np.deg2rad(90)
        # rotating a standing cylinder around the y axis, puts it flat onto the x axis

        with self.output_handler():
            xml_path = get_scene(robot_model)

            if self.new_backlash is not None:
                robot_file = self.update_backlash(xml_path)
            else:
                robot_file = URDF(xml_path, force_recompile=True).get_path()

            robot_id = p.loadURDF(
                robot_file, startPos, startOrientation, useFixedBase=1)
            self.robots.append(robot_id)

            if self.debug:
                print(robot_model)
                for i in range(p.getNumJoints(robot_id)):
                    print(p.getJointInfo(robot_id, i))

        return robot_id

    def load_backlash(self, robot_id, backlashes):
        for bl in backlashes:
            assert len(bl) == 3

            cid = p.createConstraint(
                robot_id,
                bl[0],
                robot_id,
                bl[1],
                p.JOINT_FIXED,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0])
            p.changeConstraint(cid, [0, 0, 0.1], maxForce=bl[2])

    def update_backlash(self, xml_path):
        backlashes = self.float2list(self.new_backlash)

        tree = et.parse(xml_path + ".xacro.xml")

        for i in range(6):
            bl_val = tree.find(
                ".//xacro:property[@name='backlash_val{}']".format(i + 1),
                NAMESPACE)
            bl_val.set('value', "{}".format(backlashes[i]))

        filename = "{}-bl{}".format(xml_path, np.around(self.new_backlash, 5))
        tree.write(filename + ".xacro.xml")
        robot_file = URDF(xml_path, force_recompile=True).get_path()
        return robot_file

    def clip_action(self, actions):
        return np.multiply(np.pi / 2 * np.clip(actions, -1, 1),
                           MOTOR_DIRECTIONS_DEFAULT)

    def float2list(self, val):
        if type(val) == type(1) or type(val) == type(1.0):
            return [val] * 6
        elif type(val) == type([]) or type(val) == type(np.array([])):
            assert len(val) == 6
            return val
        else:
            raise Exception(
                "the value '{}' should either be float, int or list but it's {}"
                .format(val, type(val)))

    def act2(self,
             actions,
             robot_id,
             max_force=None,
             max_vel=None,
             positionGain=None):
        actions_clipped = self.clip_action(actions)
        if self.heavy and positionGain is None:
            positionGain = .2

        if max_force is None:
            max_force = MAX_FORCE["default" if not self.heavy else "heavy"]

        if max_vel is None:
            max_vel = MAX_VEL["default" if not self.heavy else "heavy"]

        force = self.float2list(max_force)
        vel = self.float2list(max_vel)
        for idx, act in enumerate(actions_clipped):
            if positionGain is None:
                p.setJointMotorControl2(
                    self.robots[robot_id],
                    self.motor_ids[idx],
                    p.POSITION_CONTROL,
                    targetPosition=act,
                    force=force[idx],
                    maxVelocity=vel[idx])
            else:
                p.setJointMotorControl2(
                    self.robots[robot_id],
                    self.motor_ids[idx],
                    p.POSITION_CONTROL,
                    targetPosition=act,
                    force=force[idx],
                    maxVelocity=vel[idx],
                    positionGain=positionGain)

    def observe(self, robot_id):
        obs = p.getJointStates(self.robots[robot_id], self.motor_ids)
        pos = [x[0] for x in obs]
        vel = [x[1] for x in obs]
        return self.normalize(np.array(pos + vel))

    def normalize(self, posvel):
        assert len(posvel) == 12
        norm_max_vel = NORM_VEL["default" if not self.heavy else "heavy"]

        pos_norm = (posvel[:6] + np.pi / 2) / np.pi
        vel_norm = (posvel[6:] + norm_max_vel) / (norm_max_vel * 2)
        posvel_norm = np.hstack((pos_norm, vel_norm))
        posvel_shifted = posvel_norm * 2 - 1
        posvel_shifted[:6] = np.multiply(posvel_shifted[:6],
                                         MOTOR_DIRECTIONS_DEFAULT)

        return posvel_shifted

    def normalize_orn(self, orn):
        orn_norm = (np.array(orn) + np.pi / 2) / np.pi
        orn_shifted = orn_norm * 2 - 1
        return orn_shifted

    def close(self):
        p.disconnect()

    def step(self):
        p.stepSimulation()

    def set(self, posvel, robot_id):
        # !IMPORTANT set != act2... if you want the robot to stay in place
        # you also have to call act2 to set the target position

        assert len(posvel) == 12
        posvel_clipped = np.array(np.clip(posvel, -1, 1)).astype(np.float64)
        posvel_clipped[:6] *= np.pi / 2
        posvel_clipped[:6] = np.multiply(posvel_clipped[:6],
                                         MOTOR_DIRECTIONS_DEFAULT)

        for i in range(6):
            p.resetJointState(
                self.robots[robot_id],
                self.motor_ids[i],
                targetValue=posvel_clipped[i],
                targetVelocity=posvel_clipped[i + 6])

    def get_hits(self, robot1=0, robot2=1, links=None):
        if robot1 is None:
            hits = p.getContactPoints()

        elif robot2 is None:
            hits = p.getContactPoints(robot1)

        # links=(14,14) for sword+shield
        elif links is None:
            hits = p.getContactPoints(robot1, robot2)
        else:
            assert len(links) == 2
            hits = p.getContactPoints(robot1, robot2, links[0], links[1])
        return hits

    def rest(self):
        for i in range(len(self.robots)):
            self.set([0] * 12, i)

    def get_tip(self, robot_id):
        tip = 13  # TODO: might wanna check this in multi-robot setups
        state = p.getLinkState(self.robots[robot_id], tip)
        pos = state[0]
        orn = self.normalize_orn(p.getEulerFromQuaternion(state[1]))
        return (pos, orn)

    def hard_reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(1 / self.frequency)
        p.setRealTimeSimulation(0)
        if not self.heavy:
            p.loadURDF("plane.urdf")
            # pass
        else:
            p.loadURDF(URDF(get_scene("plane-big.urdf.xml")).get_path())

    def set_text(self, text=None):
        if self.debug_text is not None:
            p.removeUserDebugItem(self.debug_text)
        if text is not None and text is not "":
            self.debug_text = p.addUserDebugText(
                text, [.1, -.1, .12], textColorRGB=[1, 0, 0], textSize=6)


class PusherRobot():

    def __init__(self, debug=False, frequency=100, silent=False):
        self.debug = debug
        self.frequency = frequency
        self.output_handler = stdout_noop

        self.rest_pos = [-.5, 1, .5, 0, 0, 0]

        if silent:
            self.output_handler = stdout_redirected

        with self.output_handler():
            if debug:
                p.connect(p.GUI)
                p.resetDebugVisualizerCamera(
                    cameraDistance=0.4,
                    cameraYaw=135,
                    cameraPitch=-35,
                    cameraTargetPosition=[0, 0.05, 0])
            else:
                p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # optional for ground

        self.robot = None
        self.motor_ids = [1, 3, 5]
        self.debug_text = None

        # # GYM env has to do this
        # self.hard_reset()

    def addModel(self, robot_model, pose=None):
        if pose is None:
            pose = [0, 0, 0, 0, 0, 0]
        startPos = pose[:3]  # RGB = xyz
        startOrientation = p.getQuaternionFromEuler(pose[3:])
        # rotated around which axis? # np.deg2rad(90)
        # rotating a standing cylinder around the y axis, puts it flat onto the x axis

        with self.output_handler():
            xml_path = get_scene(robot_model)

            robot_file = URDF(xml_path, force_recompile=True).get_path()

            robot_id = p.loadURDF(
                robot_file, startPos, startOrientation, useFixedBase=1)
            self.robot = robot_id

            if self.debug:
                print(robot_model)
                for i in range(p.getNumJoints(robot_id)):
                    print(p.getJointInfo(robot_id, i))

        return robot_id

    def clip_action(self, actions):
        assert len(actions) == 3
        return np.multiply(np.pi / 2 * np.clip(actions, -1, 1),
                           MOTOR_DIRECTIONS_PUSHER)

    def float2list(self, val):
        if type(val) == type(1) or type(val) == type(1.0):
            return [val] * 3
        elif type(val) == type([]) or type(val) == type(np.array([])):
            assert len(val) == 3
            return val
        else:
            raise Exception(
                "the value '{}' should either be float, int or list but it's {}"
                .format(val, type(val)))

    def act(self, actions, max_force=None, max_vel=None, positionGain=None):
        actions = np.array(actions) + np.array(self.rest_pos[:3])
        actions_clipped = self.clip_action(actions)

        if max_force is None:
            max_force = MAX_FORCE["default"]

        if max_vel is None:
            max_vel = MAX_VEL["default"]

        force = self.float2list(max_force)
        vel = self.float2list(max_vel)
        for idx, act in enumerate(actions_clipped):
            if positionGain is None:
                p.setJointMotorControl2(
                    self.robot,
                    self.motor_ids[idx],
                    p.POSITION_CONTROL,
                    targetPosition=act,
                    force=force[idx],
                    maxVelocity=vel[idx])
            else:
                p.setJointMotorControl2(
                    self.robot,
                    self.motor_ids[idx],
                    p.POSITION_CONTROL,
                    targetPosition=act,
                    force=force[idx],
                    maxVelocity=vel[idx],
                    positionGain=positionGain)

    def observe(self):
        obs = p.getJointStates(self.robot, self.motor_ids)
        pos = [x[0] for x in obs]
        vel = [x[1] for x in obs]
        return self.normalize(np.array(pos + vel))

    def normalize(self, posvel):
        assert len(posvel) == 6
        norm_max_vel = NORM_VEL["default"]

        pos_norm = (posvel[:3] + np.pi / 2) / np.pi
        vel_norm = (posvel[3:] + norm_max_vel) / (norm_max_vel * 2)
        posvel_norm = np.hstack((pos_norm, vel_norm))
        posvel_shifted = posvel_norm * 2 - 1
        posvel_shifted[:3] = np.multiply(posvel_shifted[:3],
                                         MOTOR_DIRECTIONS_PUSHER)

        return posvel_shifted

    def close(self):
        p.disconnect()

    def step(self):
        p.stepSimulation()

    def set(self, posvel):
        # !IMPORTANT set != act2... if you want the robot to stay in place
        # you also have to call act2 to set the target position

        assert len(posvel) == 6
        posvel_clipped = np.array(np.clip(posvel, -1, 1)).astype(np.float64)
        posvel_clipped[:3] *= np.pi / 2
        posvel_clipped[:3] = np.multiply(posvel_clipped[:3],
                                         MOTOR_DIRECTIONS_PUSHER)

        for i in range(3):
            p.resetJointState(
                self.robot,
                self.motor_ids[i],
                targetValue=posvel_clipped[i],
                targetVelocity=posvel_clipped[i + 3])

    def rest(self):
        self.set(self.rest_pos)

    def hard_reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(1 / self.frequency)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")
        self.addModel("ergojr-pusher")


if __name__ == '__main__':

    pusher = PusherRobot(debug=True)
    pusher.rest()
    pusher.act(pusher.rest_pos[:3])
    for _ in range(20 * 100):
        pusher.step()
        time.sleep(1 / 100)
