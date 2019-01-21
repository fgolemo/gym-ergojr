import os

from gym.envs.registration import register

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_scene(name):
    return os.path.join(_ROOT, "scenes", name)


for headlessness in ["Graphical", "Headless"]:
    headlessness_switch = True if headlessness == "Headless" else False

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoFightStaticEnv',
        timestep_limit=1000,
        reward_threshold=150,
        kwargs={'headless': headlessness_switch, 'scaling': 0.5},
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-Plus-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
        kwargs={'base_env_id': 'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0'},
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-Plus-Half-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
        kwargs={'base_env_id': 'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0', "scaling": 0.5},
    )

    register(

        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-Plus-Training-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusTrainingEnv',
        kwargs={'base_env_id': 'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0'},
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-NoSim-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
        kwargs={'base_env_id': 'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0', "noSim": True},
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-PlusGP-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusGPEnv',
        kwargs={'base_env_id': 'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0'},
    )

    register(
        id='ErgoReacher-{}-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch}
    )

    register(
        id='ErgoReacher-{}-Halfsphere-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'goal_halfsphere': True}
    )

    register(
        id='ErgoReacher-{}-Simple-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True}
    )

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'goal_halfsphere': True}
    )

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'goal_halfsphere': True}
    )

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.0bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'goal_halfsphere': True, 'backlash': .001}
    )

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.2bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'goal_halfsphere': True, 'backlash': .2}
    )

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.4bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'goal_halfsphere': True, 'backlash': .4}
    )

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.6bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'goal_halfsphere': True, 'backlash': .6}
    )

    register(
        id='ErgoReacher-{}-DoubleGoal-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        timestep_limit=200,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'goal_halfsphere': True, 'double_goal': True}
    )

    register(
        id='ErgoReacher-{}-Simple-Backlash-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        timestep_limit=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch, 'simple': True, 'backlash': True}
    )

    register(
        id='ErgoReacher-{}-Simple-Plus-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherPlusEnv',
        kwargs={'base_env_id': 'ErgoReacher-Headless-Simple-v1'},
    )

    register(
        id='ErgoReacher-{}-Simple-Plus-v2'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherPlus2Env',
        kwargs={'base_env_id': 'ErgoReacher-Headless-Simple-v1'},
    )

    register(
        id='ErgoReacher-{}-Simple-Plus-NoSim-v2'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherPlus2Env',
        kwargs={'base_env_id': 'ErgoReacher-Headless-Simple-v1', 'nosim': True},
    )
