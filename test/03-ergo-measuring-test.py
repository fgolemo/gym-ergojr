import time

from gym_ergojr.sim.single_robot import SingleRobot

robot = SingleRobot(debug=True)

actions = [
    [-1, 0, 0, -1, 0, -1],
    [1, 0, 0, 1, 0, .2],
    [0, 0, 0, 0, 0, 0]
]
for action in actions:
    robot.act2(action) # don't need to call this at every step
    for i in range(100):
        robot.step()
        print(robot.observe().round(2))
        time.sleep(.01)

    time.sleep(1)
    print("switch")
