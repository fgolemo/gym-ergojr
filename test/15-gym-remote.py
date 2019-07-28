import gym
import gym_ergojr
import numpy as np
import tkinter as tk


class Debugger(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # env = gym.make("ErgoReacher-Graphical-Simple-Plus-v2")
        # env = gym.make("ErgoReacher-Graphical-Simple-Halfdisk-Heavy-v1")
        # env = gym.make("ErgoReacher-Graphical-DoubleGoal-v1")
        # env = gym.make("ErgoReacher-Graphical-DoubleGoal-Easy-0.5bl-7000g-v1")
        # self.env = gym.make("ErgoReacher-Graphical-Simple-Halfdisk-v1")
        self.env = gym.make("ErgoGripper-Square-JustTouch-Graphical-v1")

        self.no_actions = 6

        self.obs = self.env.reset()
        self.rew = 0
        self.don = False

        self.motors = []
        for i in range(6):
            m = tk.Scale(
                self, from_=-1, to=1, orient=tk.HORIZONTAL, resolution=0.1)
            m.pack()
            self.motors.append(m)

        tk.Button(self, text='step 1', command=self.step1).pack()
        tk.Button(self, text='step 10', command=self.step10).pack()
        tk.Button(self, text='step 100', command=self.step100).pack()
        tk.Button(self, text='reset', command=self.reset).pack()
        tk.Button(self, text='observe', command=self.observe).pack()
        tk.Button(self, text='set', command=self.sett).pack()

    def getActions(self):
        action = [m.get() for m in self.motors]

        if self.no_actions == 4:
            return [action[1], action[2], action[4], action[5]]
        return action

    def stepN(self, N):
        action = self.getActions()
        for i in range(N):
            # (_, self.obs), self.rew, self.don, _ = self.env.step(action)
            self.obs, self.rew, self.don, _ = self.env.step(action)

        print(np.around(self.obs, 3), np.around(self.rew, 3))

    def step1(self):
        self.stepN(1)

    def step10(self):
        self.stepN(10)

    def step100(self):
        self.stepN(100)

    def reset(self):
        self.env.reset()

    def observe(self):
        # print(env.unwrapped._get_obs(), env.unwrapped._getReward())
        print(np.around(self.obs, 3), np.around(self.rew, 3))

    def sett(self):
        action = self.getActions()
        self.env.unwrapped._set_state(action + [0] * self.no_actions)


root = tk.Tk()
Debugger(root).pack(side="top", fill="both", expand=True)
root.mainloop()
