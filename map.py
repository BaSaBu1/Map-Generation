import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from lloyd import Field

sys.path.append(os.path.join(os.path.dirname(__file__), 'Delaunator-Python'))
from Delaunator import Delaunator


class Map:
    def __init__(self, p, size=1):
        self.grid_size = size
        
        # Perform Lloyd's relaxation
        self.points = lloyd(p, 2)
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
                                        self.points[self.triangles[i + 2]]) for i in range(0, len(self.triangles), 3)])
        
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
                        'b-', linewidth=1)

    
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