from gym_ergojr.sim.abstract_robot import AbstractRobot


class SingleRobot(AbstractRobot):

    def __init__(self, robot_model="ergojr-penholder", debug=False, frequency=100):
        super().__init__(debug, frequency)
        self.addModel(robot_model)

    def act(self, actions, **kwargs):
        super().act(actions, 0)

    def act2(self, actions, **kwargs):
        super().act2(actions, 0)

    def observe(self, **kwargs):
        return super().observe(0)

    def set(self, posvel, **kwargs):
        super().set(posvel, 0)

    def reset(self):
        self.set([0]*12)

    def get_tip(self):
        return super().get_tip(0)


