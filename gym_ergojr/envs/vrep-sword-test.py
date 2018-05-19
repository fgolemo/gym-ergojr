import os
import time

from vrepper.core import vrepper

venv = vrepper(headless=False)
venv.start()
venv.load_scene(os.getcwd() + '/../scenes/poppy_ergo_jr_fight_sword1.ttt')
venv.start_blocking_simulation()
cam = venv.get_object_by_name('cam', is_joint=False)

motors1 = []
for i in range(6):
    motor = venv.get_object_by_name('r1m{}'.format(i + 1), is_joint=True)
    motors1.append(motor)

motors2 = []
for i in range(6):
    motor = venv.get_object_by_name('r2m{}'.format(i + 1), is_joint=True)
    motors2.append(motor)

config = [-45, 40, -30, 0, 0, 0]
sword_collision = venv.get_collision_object("sword_hit")

for i, m in enumerate(motors1):
    m.set_position_target(config[i])

for _ in range(20):
    venv.step_blocking_simulation()
    time.sleep(0.1)

# time.sleep(2)

config = [-45, 40, -30, -50, 0, 0]
for i, m in enumerate(motors1):
    m.set_position_target(config[i])

for _ in range(30):
    venv.step_blocking_simulation()
    time.sleep(0.1)

time.sleep(2)

# def reset():
#     venv.stop_simulation()
#     venv.start_nonblocking_simulation()
#
#     print("SETTING MOTORS")
#
#     for i, m in enumerate(motors):
#         m.set_position_target(restMotors[i])
#
#     print("RESETTING BALL")
#
#     ball.set_position(0, 0.05, .3)
#
#     # test = venv.call_script_function('spawnBall', params)
#     # print(test)
#     time.sleep(1)
#     print("MAKING SIM SYNC")
#
#     venv.make_simulation_synchronous(True)
#
#     print('(Ergo ball scene) initialized')
#
#
# throwStart = [0, 40, 40, 0, -90, -55]
# throwEnd = [0, -50, -40, 0, -50, -30]
#
#
# def gotoPos(pos):
#     for i, m in enumerate(motors):
#         m.set_position_target(pos[i])
#
#
# for i in range(2):
#     reset()
#     gotoPos(throwStart)  # start pos
#     for _ in range(30):
#         venv.step_blocking_simulation()
#         # print(ball.get_position(), venv.check_collision(ball_collision))
#         print(ball.get_position(), ball_collision.is_colliding())
#     gotoPos(throwEnd)  # end pos
#     forces = motors[0].get_joint_force()
#     print(forces)
#     for _ in range(30):
#         venv.step_blocking_simulation()
#         print(ball.get_position(), ball_collision.is_colliding())
#     time.sleep(1)
