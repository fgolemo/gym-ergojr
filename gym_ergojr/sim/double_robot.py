from gym_ergojr.sim.abstract_robot import AbstractRobot
import numpy as np


class DoubleRobot(AbstractRobot):

    def __init__(self,
                 robot_model1="ergojr-sword",
                 robot_model2="ergojr-shield",
                 debug=False,
                 frequency=100
                 ):
        super().__init__(debug, frequency)
        self.addModel(robot_model1, pose=[0, 0, 0, 0, 0, 0])
        self.addModel(robot_model2, pose=[0, .37, 0, 0, 0, np.pi])

    def observe_both(self):
        return np.hstack((self.observe(0), self.observe(1)))

    def get_hits(self, links=None, **kwargs):
        return super().get_hits(self.robots[0], self.robots[1], links)


if __name__ == '__main__':
    import time

    d = DoubleRobot(debug=True)
    d.rest()

    d.step()
    d.act2([1, 0, 0, 1, 0, 0], 0)
    d.act2([-1, 0, 0, -1, 0, 0], 1)
    for i in range(10):
        print(d.observe_both().round(2))
        d.step()

    time.sleep(2)
    d.close()
