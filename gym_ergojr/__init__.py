import os

from gym.envs.registration import register

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_scene(name):
    return os.path.join(_ROOT, "scenes", name)


register(
    id='ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0',
    entry_point='gym_ergojr.envs:ErgoFightStaticEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'headless': False, 'scaling': 0.5},
)

register(
    id='ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0',
    entry_point='gym_ergojr.envs:ErgoFightStaticEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'headless': True, 'scaling': 0.5},
)

register(
    id='ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-Plus-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0'},
)

register(
    id='ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0'},
)

register(
    id='ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-Half-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0', "scaling": 0.5},
)

register(

    id='ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-Training-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusTrainingEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0'},
)

register(
    id='ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-NoSim-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0', "noSim": True},
)

register(
    id='ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-NoSim-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0', "noSim": True},
)

register(
    id='ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-PlusGP-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusGPEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0'},
)

register(
    id='ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-PlusGP-v0',
    entry_point='gym_ergojr.envs:ErgoFightPlusGPEnv',
    kwargs={'base_env_id': 'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0'},
)

register(
    id='ErgoReacher-Headless-v0',
    entry_point='gym_ergojr.envs:ErgoReacherEnv',
    timestep_limit=100,
    reward_threshold=0,
    kwargs={'headless': True}
)

register(
    id='ErgoReacher-Graphical-v0',
    entry_point='gym_ergojr.envs:ErgoReacherEnv',
    timestep_limit=100,
    reward_threshold=0,
    kwargs={'headless': False}
)

register(
    id='ErgoReacher-Headless-Simple-v1',
    entry_point='gym_ergojr.envs:ErgoReacherEnv',
    timestep_limit=100,
    reward_threshold=0,
    kwargs={'headless': True, 'simple': True}
)

register(
    id='ErgoReacher-Graphical-Simple-v1',
    entry_point='gym_ergojr.envs:ErgoReacherEnv',
    timestep_limit=100,
    reward_threshold=0,
    kwargs={'headless': False, 'simple': True}
)
