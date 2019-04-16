import gym
import gym_ergojr
from gym_ergojr.utils.kinematics import inverse_kinematics

env = gym.make("ErgoReacher-Graphical-Simple-Halfdisk-v1")


for _ in range(100):
    obs = env.reset()

    while True:
        # get the angles of the joints that would move the tip of the robot to the goal
        angles = inverse_kinematics(env.unwrapped.goal, 100)

        # the angles are in [-90,90] but they need to be in [-1,1], and also we don't need joint0 and joint3 in this task
        action = angles[[1,2,4,5]]/90

        obs, rew, done, misc = env.step(action)
        if done:
            break

