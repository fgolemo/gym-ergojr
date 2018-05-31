from tkinter import *
import gym
import gym_ergojr
import numpy as np

env = gym.make("ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-Plus-v0")
env.reset()


def getActions():
    return [m.get() for m in motors]


def stepN(N):
    action = getActions()
    print("action", action)
    for i in range(N):
        obs, _, _, _ = env.step(action)
    print(np.around(obs, 2))


def step1():
    stepN(1)


def step10():
    stepN(10)


def step100():
    stepN(100)


def reset():
    env.reset()
    for m in motors:
        m.set(0)


master = Tk()

motors = []
for i in range(6):
    m = Scale(master, from_=-1, to=1, orient=HORIZONTAL, resolution=0.1)
    m.pack()
    motors.append(m)

Button(master, text='step 1', command=step1).pack()
Button(master, text='step 10', command=step10).pack()
Button(master, text='step 100', command=step100).pack()
Button(master, text='reset', command=reset).pack()

mainloop()
