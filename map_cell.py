import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from helper import *
sys.path.append(os.path.join(os.path.dirname(__file__), 'Delaunator-Python'))



from Delaunator import Delaunator

N = 625
WIDTH = 25
WATER_LEVEL = 0.5

# Plotting
fig, ax = plt.subplots()

# Generate random points
p = np.random.rand(N, 2) * WIDTH

# Perform Lloyd's relaxation
p1 = lloyd(p, 2)

# Compute Delaunay triangulation
delaunay = Delaunator(p1)
del_tri = delaunay.triangles
halfedges = delaunay.halfedges

# Calculate circumcenters of the triangles
centers = np.array([
    circumCenter(
        p1[del_tri[i]],
        p1[del_tri[i + 1]],
        p1[del_tri[i + 2]]
    )
    for i in range(0, len(del_tri), 3)
])


# Elevation now per site (Voronoi cell), not per triangle center
elevation = assignElevation(WATER_LEVEL, p1, WIDTH)


def color_fn(r: int) -> str:
    return 'royalblue' if elevation[r] < WATER_LEVEL else 'olivedrab'

seen = set()

for e in range(len(halfedges)):
    r = del_tri[next_halfedge(e)]  # site index for this cell
    if r in seen:
        continue
    seen.add(r)

    ring_edges = edges_around_point(halfedges, del_tri, e)

    # Build polygon from circumcenters of adjacent triangles
    poly = []
    open_cell = False
    for he in ring_edges:
        t = triangle_of_edge(he)
        poly.append(centers[t])
        # Detect hull adjacency (open cell) if any outgoing is -1
        outgoing = next_halfedge(he)
        if halfedges[outgoing] == -1:
            open_cell = True

    # Skip open cells touching the convex hull
    if open_cell or len(poly) < 3:
        continue

    poly = np.array(poly)

    ax.fill(poly[:, 0], poly[:, 1], color=color_fn(r), alpha=0.6, edgecolor='none')

# Plot the original points
ax.scatter(p1[:, 0], p1[:, 1], color='black', s=5, zorder=5)

# Plot Delaunay triangles
# for i in range(0, len(del_tri), 3):
#     triangle = [
#         p1[del_tri[i]],
#         p1[del_tri[i+1]],
#         p1[del_tri[i+2]],
#         p1[del_tri[i]]
#     ]
#     triangle = np.array(triangle)
#     ax.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.5)

# Plot Voronoi edges
# for e in range(len(halfedges)):
#     if halfedges[e] != -1 and e < halfedges[e]:
#         t1 = e // 3  # Triangle index for current halfedge
#         t2 = halfedges[e] // 3  # Triangle index for opposite halfedge
        
#         # Draw edge between circumcenters
#         ax.plot([centers[t1][0], centers[t2][0]], 
#                 [centers[t1][1], centers[t2][1]], 
#                 'b-', linewidth=1)

ax.set_aspect('equal')
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, WIDTH)
plt.show()
