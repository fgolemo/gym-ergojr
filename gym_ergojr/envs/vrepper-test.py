import os
import time

from vrepper.core import vrepper

venv = vrepper(headless=False)
venv.start()
venv.load_scene(os.getcwd() + '/../scenes/poppy_ergo_jr_vanilla_ball.ttt')

motors = []
for i in range(6):
    motor = venv.get_object_by_name('m{}'.format(i + 1), is_joint=True)
    motors.append(motor)

restMotors = [0, -90, 35, 0, 55, -90]


def reset():
    venv.stop_simulation()
    venv.start_nonblocking_simulation()

    print("SETTING MOTORS")

    for i, m in enumerate(motors):
        m.set_position_target(restMotors[i])

    print("SPAWNING BALL")

    params = venv.create_params(
        [],
        [.035, .035, .035,  # size
         0, 0.05, .3,  # position
         .01],  # weight)
        [],
        ''
    )
    test = venv.call_script_function('spawnBall', params)
    print(test)
    time.sleep(.5)
    print("MAKING SIM SYNC")

    venv.make_simulation_synchronous(True)

    print('(Ergo ball scene) initialized')


throwStart = [0, 40, 40, 0, -90, -55]
throwEnd = [0, -50, -40, 0, -50, -30]


def gotoPos(pos):
    for i, m in enumerate(motors):
        m.set_position_target(pos[i])


for i in range(2):
    reset()
    ball = venv.get_object_by_name("ball")
    gotoPos(throwStart)  # start pos
    for _ in range(30):
        venv.step_blocking_simulation()
        print(ball.get_position())
    gotoPos(throwEnd)  # end pos
    forces = motors[0].get_joint_force()
    print(forces)
    for _ in range(30):
        venv.step_blocking_simulation()
        print(ball.get_position())
    time.sleep(1)

venv.stop_blocking_simulation()
venv.end()
