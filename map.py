import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from lloyd import Field
from noise import pnoise2

sys.path.append(os.path.join(os.path.dirname(__file__), 'Delaunator-Python'))
from Delaunator import Delaunator


class Map:
    def __init__(self, p, size=1, water_level=0.3, noise_scale=4, cluster=10):
        self.grid_size = size
        self.water_level = water_level
        self.noise_scale = noise_scale
        self.cluster = cluster
        
        # Perform Lloyd's relaxation
        self.points = lloyd(p, 3)
        self.numRegions = len(self.points)
        
        # Compute Delaunay triangulation
        delaunay = Delaunator(self.points)
        self.triangles = delaunay.triangles
        self.numTriangles = len(self.triangles)
        self.halfedges = delaunay.halfedges
        self.numEdges = len(self.halfedges)
        
        # Calculate circumcenters of the triangles (Voronoi vertices)
        self.centers = np.array([circumCenter(self.points[self.triangles[i]], 
                                        self.points[self.triangles[i + 1]],
                                        self.points[self.triangles[i + 2]]) 
                                for i in range(0, len(self.triangles), 3)])
        
        self.assignAltitudes()
        
    def plotPoints(self, ax):
        ax.scatter(self.points[:, 0], self.points[:, 1], color='black', s=5)
        
    def plotDelaunay(self, ax):
        for i in range(0, self.numTriangles, 3):
            triangle = [
                self.points[self.triangles[i]],
                self.points[self.triangles[i+1]],
                self.points[self.triangles[i+2]],
                self.points[self.triangles[i]]
            ]
            triangle = np.array(triangle)
            ax.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.5)
            
    def plotVoronoi(self, ax):
        for e in range(self.numEdges):
            if self.halfedges[e] != -1 and e < self.halfedges[e]:
                t1 = e // 3  # Triangle index for current halfedge
                t2 = self.halfedges[e] // 3  # Triangle index for opposite halfedge
                
                # Draw edge between circumcenters
                ax.plot([self.centers[t1][0], self.centers[t2][0]], 
                        [self.centers[t1][1], self.centers[t2][1]], 
                        'b-', linewidth=1, color='gray', alpha=0.5)

    def assignAltitudes(self):
        # Generate random centers for islands
        island_centers = np.random.uniform(0.2, 0.8, (int(self.cluster), 2)) * self.grid_size
        
        # Calculate distances to the nearest center
        distances = np.full(self.numRegions, np.inf)
        for center in island_centers:
            d = np.linalg.norm(self.points - center, axis=1)
            distances = np.minimum(distances, d)
        
        # Normalize distances to [0, 1]
        normalized_dists = distances / (self.grid_size / 2) 

        self.altitudes = np.zeros(self.numRegions)
        
        for i in range(self.numRegions):
            # Scale coordinates for noise
            nx = self.points[i, 0] / self.grid_size * self.noise_scale
            ny = self.points[i, 1] / self.grid_size * self.noise_scale

            # Generate Perlin noise
            noise_val = (pnoise2(nx, ny, octaves=6) + 1) / 2
            
            # Calculate penalty
            penalty = 1.0 * (normalized_dists[i] ** 2)
            
            # Altitude calculation
            self.altitudes[i] = noise_val - penalty
            
    
    def plotLand(self, ax):
        # Build index map
        index = np.full(self.numRegions, -1, dtype=int)
        for e in range(self.numEdges):
            start_point = self.triangles[e]
            if index[start_point] == -1:
                index[start_point] = e
                
        for i in range(self.numRegions):
            if index[i] == -1:
                continue
                
            # Traverse Voronoi region
            vertices = []
            e0 = index[i]
            e = e0
            
            while True:
                t = e // 3
                vertices.append(self.centers[t])
                
                # Move to next edge around point i
                prev_e = e - 1 if e % 3 != 0 else e + 2
                opp_e = self.halfedges[prev_e]
                
                if opp_e == -1:
                    break
                    
                e = opp_e
                if e == e0:
                    break
            
            if len(vertices) > 2:
                vertices = np.array(vertices)
                color = 'forestgreen' if self.altitudes[i] > self.water_level else 'deepskyblue'
                ax.fill(vertices[:, 0], vertices[:, 1], color=color)
        
        
def lloyd(points, iterations=1):
    """Perform Lloyd's relaxation on a set of points.
    
    Args:
        points (np.ndarray): An array of shape (N, 2) representing the initial points.
        iterations (int): The number of relaxation iterations to perform.   
        
    Returns:
        np.ndarray: The relaxed points after the specified number of iterations.    
    """
    field = Field(points)
    for i in range(iterations):
        field.relax()
        
    return field.get_points()

def circumCenter(a, b, c):
    """Calculate the circumcenter of a triangle given by points a, b, and c.
    
    Args:
        a, b, c (np.ndarray): Arrays of shape (2,) representing the triangle vertices.
        
    Returns:
        np.ndarray: An array of shape (2,) representing the circumcenter of the triangle.
    """
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
    uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
    return np.array([ux, uy])

