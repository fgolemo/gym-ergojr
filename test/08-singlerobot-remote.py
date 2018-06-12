from tkinter import *
from gym_ergojr.sim.single_robot import SingleRobot

master = Tk()

robot = SingleRobot(debug=True)


def getActions():
    return [m.get() for m in motors]


def stepN(N):
    action = getActions()
    print("action", action)
    robot.act2(action)
    for i in range(N):
        robot.step()
    print("tip", robot.get_tip())


def step1():
    stepN(1)


def step10():
    stepN(10)


def step100():
    stepN(100)


def reset():
    robot.reset()
    robot.step()
    robot.reset()
    robot.step()


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
