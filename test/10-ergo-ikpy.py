import ikpy
import numpy as np
from ikpy import plot_utils


my_chain = ikpy.chain.Chain.from_urdf_file("urdfs/PoppyErgoJr_pen.urdf.xml")

target_vector = [ -0.05, -0.05, 0.2]
target_frame = np.eye(4) # means don't care about orientation
target_frame[:3, 3] = target_vector

print("The angles of each joints are : ", my_chain.inverse_kinematics(target_frame))

real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_frame))
print("Computed position vector : %s, original position vector : %s" % (real_frame[:3, 3], target_frame[:3, 3]))

import matplotlib.pyplot as plt
ax = plot_utils.init_3d_figure()
my_chain.plot(my_chain.inverse_kinematics(target_frame), ax, target=target_vector)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
ax.set_zlim(0, .3)
plt.show()

