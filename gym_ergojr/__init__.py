import os

from gym.envs.registration import register

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_scene(name):
    return os.path.join(_ROOT, "scenes", name)


for headlessness in ["Graphical", "Headless"]:
    headlessness_switch = True if headlessness == "Headless" else False

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-v0'.format(
            headlessness),
        entry_point='gym_ergojr.envs:ErgoFightStaticEnv',
        max_episode_steps=1000,
        reward_threshold=150,
        kwargs={
            'headless': headlessness_switch,
            'scaling': 0.5
        },
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-Plus-v0'.format(
            headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
        kwargs={
            'base_env_id':
                'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0'
        },
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-Plus-Half-v0'.format(
            headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
        kwargs={
            'base_env_id':
                'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0',
            "scaling":
                0.5
        },
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-Plus-Training-v0'
        .format(headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusTrainingEnv',
        kwargs={
            'base_env_id':
                'ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0'
        },
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-NoSim-v0'.format(
            headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusEnv',
        kwargs={
            'base_env_id':
                'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0',
            "noSim":
                True
        },
    )

    register(
        id='ErgoFightStatic-{}-Shield-Move-HalfRand-Bullet-PlusGP-v0'.format(
            headlessness),
        entry_point='gym_ergojr.envs:ErgoFightPlusGPEnv',
        kwargs={
            'base_env_id':
                'ErgoFightStatic-Graphical-Shield-Move-HalfRand-Bullet-v0'
        },
    )

    register(
        id='ErgoReacher-{}-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={'headless': headlessness_switch})

    register(
        id='ErgoReacher-{}-Halfsphere-v0'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'goal_halfsphere': True
        })

    register(
        id='ErgoReacher-{}-Simple-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True
        })

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True
        })

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True
        })

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.0bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'backlash': .001
        })

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.2bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'backlash': .2
        })

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.4bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'backlash': .4
        })

    register(
        id='ErgoReacher-{}-Simple-Halfdisk-Heavy-0.6bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'backlash': .6
        })

    register(
        id='ErgoReacher-{}-DoubleGoal-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=200,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'double_goal': True
        })

    register(
        id='ErgoReacher-{}-DoubleGoal-0.4bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=200,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'double_goal': True,
            'backlash': .4
        })

    register(
        id='ErgoReacher-{}-DoubleGoal-0.5bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=200,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'double_goal': True,
            'backlash': .5
        })

    register(
        id='ErgoReacher-{}-DoubleGoal-Easy-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=200,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': False,
            'double_goal': True
        })

    register(
        id='ErgoReacher-{}-DoubleGoal-Easy-0.4bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=200,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': False,
            'double_goal': True,
            'backlash': .4
        })

    register(
        id='ErgoReacher-{}-DoubleGoal-Easy-0.5bl-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=200,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': False,
            'double_goal': True,
            'backlash': .5
        })

    register(
        id='ErgoReacher-{}-DoubleGoal-Easy-0.5bl-7000g-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherHeavyEnv',
        max_episode_steps=200,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': False,
            'double_goal': True,
            'backlash': .5,
            'max_force': 7000
        })

    register(
        id='ErgoReacher-{}-Simple-Backlash-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'backlash': True
        })

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
        kwargs={
            'base_env_id': 'ErgoReacher-Headless-Simple-v1',
            'nosim': True
        },
    )

    register(
        id='ErgoReacher-{}-MultiGoal-Halfdisk-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        max_episode_steps=300,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'simple': True,
            'goal_halfsphere': True,
            'multi_goal': True,
            'goals': 3
        })

    register(
        id='ErgoReacher-{}-Gripper-MobileGoal-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoReacherEnv',
        max_episode_steps=100,
        reward_threshold=0,
        kwargs={
            'headless': headlessness_switch,
            'goal_halfsphere': True,
            'gripper': True
        })

    register(
        id='ErgoPusher-{}-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoPusherEnv',
        max_episode_steps=100,
        kwargs={'headless': headlessness_switch})

    register(
        id='ErgoGripper-Linear-{}-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoGripperEnv',
        max_episode_steps=100,
        kwargs={
            'headless': headlessness_switch,
            'cube_spawn': "linear"
        })

    register(
        id='ErgoGripper-Square-{}-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoGripperEnv',
        max_episode_steps=100,
        kwargs={
            'headless': headlessness_switch,
            'cube_spawn': "square"
        })

    register(
        id='ErgoGripper-Square-JustTouch-{}-v1'.format(headlessness),
        entry_point='gym_ergojr.envs:ErgoGripperEnv',
        max_episode_steps=100,
        kwargs={
            'headless': headlessness_switch,
            'cube_spawn': "square",
            'touchy': True
        })
