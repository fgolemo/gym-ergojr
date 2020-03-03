import gym
import gym_ergojr

env = gym.make("ErgoReacherAugmented-Graphical-MultiGoal-Halfdisk-Long-v2")
env.reset()

for _ in range(100):
    _, rew, _, _ = env.step(env.action_space.sample())
    print(rew)