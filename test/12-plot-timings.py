import matplotlib.pyplot as plt
import numpy as np


d = np.load("../gym_ergojr/envs/timings.npz")
timings = d["time"]
print (timings.shape)

plt.plot(np.arange(0,len(timings)),timings)
plt.show()

