import numpy as np
import matplotlib.pyplot as plt
from map import Map

N = 200
WIDTH = 25

# Generate random points
p = np.random.rand(N, 2) * WIDTH

# Create map
map = Map(p, WIDTH)

# Plotting
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, WIDTH)

map.plotPoints(ax)
map.plotVoronoi(ax)

plt.show()
