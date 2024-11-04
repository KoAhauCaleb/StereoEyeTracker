import numpy as np
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

points = [(1, 7.5), (2, 10), (2, 4.6), (12, 3.2), (10, 2.5), (10, 11), (12, 10)]

a_points = np.array(points)
x = a_points[:, 0]
y = a_points[:, 1]

ell = EllipseModel()
ell.estimate(a_points)

xc, yc, a, b, theta = ell.params

print("center = ", (xc, yc))
print("angle of rotation = ", theta)
print("axes = ", (a, b))

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].scatter(x, y)

axs[1].scatter(x, y)
axs[1].scatter(xc, yc, color='red', s=100)
axs[1].set_xlim(0, 15)
axs[1].set_ylim(0, 15)

ell_patch = Ellipse((xc, yc), 2 * a, 2 * b, angle=theta * 180 / np.pi, edgecolor='red', facecolor='none')

axs[1].add_patch(ell_patch)
plt.show()
