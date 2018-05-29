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

if __name__ == '__main__':
    import time

    d = DoubleRobot(debug=True)
    time.sleep(2)
    d.close()
