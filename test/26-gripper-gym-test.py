import gym
import gym_ergojr

env = gym.make("ErgoGripper-Headless-v1")

for i in range(5):
    env.reset()
    done = False

    print("\n\n=== RESET ===\n\n")

    while not done:
        action = env.action_space.sample()

        (img, obs), rew, done, misc = env.step(action)

        env.render("human")

        print(img.shape, obs, rew, done, misc)
