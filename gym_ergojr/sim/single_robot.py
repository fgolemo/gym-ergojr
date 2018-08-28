from gym_ergojr.sim.abstract_robot import AbstractRobot


class SingleRobot(AbstractRobot):

    def __init__(self, robot_model="ergojr-penholder", debug=False, frequency=100, backlash=False):
        self.robot_model = robot_model
        super().__init__(debug, frequency, backlash)
        self.hard_reset()

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

    def hard_reset(self):
        super().hard_reset()
        for _ in range(5):
            self.id = self.addModel(self.robot_model)
            if self.id < 0:
                continue
            else:
                if self.backlash:
                    self.load_backlash(self.id, [
                        (5,6,.4),
                        (11,12,.2)
                    ])
                return

        print("couldn't load URDF after 5 attempts:", self.robot_model)


if __name__ == '__main__':
    import numpy as np
    r = SingleRobot(debug=True)
    r.act2([0,0,0,1,0,0])

    for i in range(300):
        r.step()
        print(np.around(r.observe()[:6],2))