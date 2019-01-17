from tkinter import *
import gym
import gym_ergojr

master = Tk()

# env = gym.make("ErgoReacher-Graphical-Simple-Plus-v2")
env = gym.make("ErgoReacher-Graphical-Simple-Halfdisk-Heavy-v1")
env.reset()


def getActions():
    action = [m.get() for m in motors]

    return [action[1],action[2],action[4],action[5]]


def stepN(N):
    action = getActions()
    for i in range(N):
        env.step(action)
    # print("tip", robot.get_tip())


def step1():
    stepN(1)


def step10():
    stepN(10)


def step100():
    stepN(100)


def reset():
    env.reset()

def observe():
    print(env.unwrapped._get_obs(), env.unwrapped._getReward())

def sett():
    action = getActions()
    env.unwrapped._set_state(action + [0]*4)

motors = []
for i in range(6):
    m = Scale(master, from_=-1, to=1, orient=HORIZONTAL, resolution=0.1)
    m.pack()
    motors.append(m)

Button(master, text='step 1', command=step1).pack()
Button(master, text='step 10', command=step10).pack()
Button(master, text='step 100', command=step100).pack()
Button(master, text='reset', command=reset).pack()
Button(master, text='observe', command=observe).pack()
Button(master, text='set', command=sett).pack()

mainloop()
