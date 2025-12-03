"""
Procedural Terrain Map Generation Module.

Generates 2D Voronoi-based terrain maps using Lloyd's relaxation and Perlin noise.

Author: Batsambuu Batbold
Date: December 2025
"""

import sys
import os

import numpy as np
from matplotlib.collections import PolyCollection
from noise import pnoise2

from lloyd import Field

sys.path.append(os.path.join(os.path.dirname(__file__), "Delaunator-Python"))
from Delaunator import Delaunator


class Map:
    """
    Procedurally generated terrain map based on Voronoi diagrams.
    
    Attributes:
        grid_size: Map dimensions.
        water_level: Altitude threshold for water vs land.
        noise_scale: Perlin noise frequency.
        cluster: Number of island clusters.
        height, depth: Color gradient ranges for land/water.
        points: Voronoi site coordinates.
        triangles: Delaunay triangulation indices.
        centers: Voronoi vertices (triangle circumcenters).
        altitudes: Terrain height per region.
    """

    def __init__(self, p, size=1, water_level=0.4, noise_scale=3, cluster=4, height=0.7, depth=0.5):
        self.grid_size = size
        self.water_level = water_level
        self.noise_scale = noise_scale
        self.cluster = cluster
        self.height = height
        self.depth = depth
        
        # Apply Lloyd's relaxation for uniform point distribution
        self.points = lloyd(p, 3)
        self.numRegions = len(self.points)
        
        # Compute Delaunay triangulation
        delaunay = Delaunator(self.points)
        self.triangles = np.array(delaunay.triangles)
        self.halfedges = np.array(delaunay.halfedges)
        self.numTriangles = len(self.triangles) // 3
        self.numEdges = len(self.halfedges)
        
        # Compute Voronoi vertices
        self.centers = self._get_circumcenters()
        
        # Compute polygon geometry
        self._build_polygons()
        
        self.assignAltitudes()

    def _build_polygons(self):
        """Compute Voronoi polygon vertices for each region."""
        index = np.full(self.numRegions, -1, dtype=int)
        for e in range(len(self.halfedges)):
            start_point = self.triangles[e]
            if index[start_point] == -1:
                index[start_point] = e
        
        self.polygons = []
        self.polygon_indices = []
        
        for i in range(self.numRegions):
            if index[i] == -1:
                continue
            
            vertices = []
            e0 = index[i]
            e = e0
            
            while True:
                t = e // 3
                vertices.append(self.centers[t])
                
                prev_e = e - 1 if e % 3 != 0 else e + 2
                opp_e = self.halfedges[prev_e]
                
                if opp_e == -1:
                    break
                
                e = opp_e
                if e == e0:
                    break
            
            if len(vertices) > 2:
                self.polygons.append(vertices)
                self.polygon_indices.append(i)

    def assignAltitudes(self):
        """Generate terrain using Perlin noise masked by distance to island centers."""
        island_centers = np.random.uniform(0.2, 0.8, (self.cluster, 2)) * self.grid_size
        
        # Vectorized distance to nearest island center
        all_dists = np.array([np.linalg.norm(self.points - c, axis=1) for c in island_centers])
        distances = np.min(all_dists, axis=0)
        normalized_dists = distances / (self.grid_size / 2)
        
        # Vectorized noise
        scaled_pts = self.points / self.grid_size * self.noise_scale
        noise_vals = np.array([(pnoise2(x, y, octaves=4) + 1) / 2 
                            for x, y in scaled_pts])
        
        self.altitudes = noise_vals - normalized_dists ** 2

    def plotLand(self, ax):
        """Render Voronoi regions as colored polygons."""
        colors = [self.get_color(self.altitudes[i]) for i in self.polygon_indices]
        pc = PolyCollection(self.polygons, facecolors=colors, edgecolors='none')
        ax.add_collection(pc)

    def plotVoronoi(self, ax):
        """Draw Voronoi cell edges."""
        for e in range(len(self.halfedges)):
            if self.halfedges[e] != -1 and e < self.halfedges[e]:
                t1 = e // 3
                t2 = self.halfedges[e] // 3
                ax.plot([self.centers[t1][0], self.centers[t2][0]], 
                        [self.centers[t1][1], self.centers[t2][1]], 
                        'b-', linewidth=1, color='gray', alpha=0.5)

    def plotDelaunay(self, ax):
        """Draw Delaunay triangulation."""
        ax.triplot(self.points[:, 0], self.points[:, 1], self.triangles, 
                'b-', linewidth=0.5, alpha=0.5)

    def _get_circumcenters(self):
        """Calculate circumcenters for all triangles."""
        tri_indices = self.triangles.reshape(-1, 3)
        
        p1 = self.points[tri_indices[:, 0]]
        p2 = self.points[tri_indices[:, 1]]
        p3 = self.points[tri_indices[:, 2]]
        
        ax, ay = p1[:, 0], p1[:, 1]
        bx, by = p2[:, 0], p2[:, 1]
        cx, cy = p3[:, 0], p3[:, 1]
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        d[d == 0] = 1.0  # Avoid division by zero for collinear points
        
        a_sq = ax**2 + ay**2
        b_sq = bx**2 + by**2
        c_sq = cx**2 + cy**2
        
        ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d
        uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d
        
        return np.column_stack((ux, uy))

    def get_color(self, alt):
        """Map altitude to RGB color (blue for water, green for land)."""
        if alt < self.water_level:
            val = np.clip((alt - (self.water_level - self.depth)) / self.depth, 0, 1)
            return (0.6 * val, 0.9 * val, 0.5 + 0.5 * val)
        else:
            val = np.clip((alt - self.water_level) / (self.height - self.water_level), 0, 1)
            return (0.7 - 0.7 * val, 0.9 - 0.5 * val, 0.7 - 0.7 * val)


def lloyd(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply Lloyd's relaxation for uniform point distribution."""
    field = Field(points)
    for _ in range(iterations):
        field.relax()
    return field.get_points()