import os

from gym.envs.registration import register

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_scene(name):
    return os.path.join(_ROOT, "scenes", name)


register(
    id='ErgoFightStatic-Graphical-Shield-Move-HalfRand-v0',
    entry_point='gym_ergojr.envs:ErgoFightStaticEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'headless': False, 'scaling': 0.5},
)

register(
    id='ErgoFightStatic-Headless-Shield-Move-HalfRand-v0',
    entry_point='gym_ergojr.envs:ErgoFightStaticEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'headless': True, 'scaling': 0.5},
)
