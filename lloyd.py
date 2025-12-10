"""
Lloyd's Relaxation Algorithm for Voronoi Diagrams.

Simplified standalone implementation based on:
https://github.com/duhaime/lloyd (MIT License)
"""

from scipy.spatial import Voronoi
import numpy as np


class Field:
    """Voronoi point field for Lloyd relaxation."""

    def __init__(self, points, constrain=True):
        """
        Initialize with 2D points.
        
        Args:
            points: Nx2 numpy array of coordinates
            constrain: Keep points within original bounds
        """
        if not isinstance(points, np.ndarray) or points.shape[1] != 2:
            raise ValueError('Points must be Nx2 numpy array')
        
        self.points = points.copy()
        self.constrain = constrain
        self.domains = self._get_domains(points)
        self._jitter_duplicates()
        self._build_voronoi()

    def _get_domains(self, arr):
        """Get bounding box of points."""
        return {
            'x': {'min': arr[:, 0].min(), 'max': arr[:, 0].max()},
            'y': {'min': arr[:, 1].min(), 'max': arr[:, 1].max()}
        }

    def _jitter_duplicates(self, epsilon=1e-9):
        """Add tiny noise to duplicate points."""
        while len(np.unique(self.points, axis=0)) < len(self.points):
            noise = np.random.randn(len(self.points), 2) * epsilon
            self.points += noise
            self._constrain_points()

    def _constrain_points(self):
        """Keep points within bounds."""
        if not self.constrain:
            return
        self.points[:, 0] = np.clip(self.points[:, 0], 
                                     self.domains['x']['min'], 
                                     self.domains['x']['max'])
        self.points[:, 1] = np.clip(self.points[:, 1], 
                                     self.domains['y']['min'], 
                                     self.domains['y']['max'])

    def _build_voronoi(self):
        """Build Voronoi diagram."""
        self.voronoi = Voronoi(self.points, qhull_options='Qbb Qc Qx')

    def _find_centroid(self, vertices):
        """Calculate polygon centroid."""
        area = 0
        cx = 0
        cy = 0
        
        for i in range(len(vertices) - 1):
            cross = (vertices[i, 0] * vertices[i+1, 1] - 
                    vertices[i+1, 0] * vertices[i, 1])
            area += cross
            cx += (vertices[i, 0] + vertices[i+1, 0]) * cross
            cy += (vertices[i, 1] + vertices[i+1, 1]) * cross
        
        area /= 2
        if abs(area) < 1e-10:
            area = 1e-10
        
        cx = cx / (6 * area)
        cy = cy / (6 * area)
        
        if self.constrain:
            cx = np.clip(cx, self.domains['x']['min'], self.domains['x']['max'])
            cy = np.clip(cy, self.domains['y']['min'], self.domains['y']['max'])
        
        return np.array([cx, cy])

    def relax(self):
        """Move points to Voronoi cell centroids."""
        centroids = []
        
        for idx in self.voronoi.point_region:
            region = [i for i in self.voronoi.regions[idx] if i != -1]
            if len(region) == 0:
                centroids.append(self.points[len(centroids)])
                continue
            
            region.append(region[0])  # Close polygon
            vertices = self.voronoi.vertices[region]
            centroids.append(self._find_centroid(vertices))
        
        self.points = np.array(centroids)
        self._constrain_points()
        self._jitter_duplicates()
        self._build_voronoi()

    def get_points(self):
        """Return current point positions."""
        return self.points.copy()
