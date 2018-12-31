from setuptools import setup

setup(name='gym_ergojr',
      version='1.1',
      install_requires=['gym>=0.2.3',
                        'pybullet>=1.9.4',
                        'sklearn',
                        'scipy',
                        "tqdm",
                        "matplotlib",
                        "numpy"
                        ]
)
