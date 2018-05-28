import time
import numpy as np
from gym_ergojr.sim.single_robot import SingleRobot

robot = SingleRobot(debug=True)

for i in range(20):
    robot.set(np.random.normal(0, 1, 12))
    print(robot.observe().round(2))
    # robot.step()
    time.sleep(.1)
    robot.act2(np.random.normal(0, 1, 6))
    robot.step()
    print(robot.observe().round(2))
    time.sleep(.5)

    print("---")

time.sleep(1)
print("switch")
