"""
Procedural Terrain Map Generation Module.

This module provides the core 2D terrain generation algorithm using computational
geometry techniques. It combines Voronoi diagrams, Lloyd's relaxation, and 
Perlin noise to create realistic procedural terrain maps with biome-based 
coloring.

Key Features:
    - Voronoi-based region partitioning using Delaunay triangulation
    - Lloyd's relaxation for uniform point distribution
    - Multi-octave Perlin noise for natural elevation and moisture patterns
    - Export utilities for heightmaps and colormaps (Blender integration)

Example:
    >>> import numpy as np
    >>> from map import Map
    >>> points = np.random.rand(1000, 2)
    >>> terrain = Map(points, water_level=0.35, noise_scale=3.0)
    >>> fig, ax = plt.subplots()
    >>> terrain.plotLand(ax)

Author: Batsambuu Batbold
Course: MATH 437 - Computational Geometry
Date: December 2025
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "Delaunator-Python"))
sys.path.append(os.path.join(os.path.dirname(__file__), "lloyd"))

import numpy as np
from matplotlib.collections import PolyCollection
from noise import pnoise2
from scipy.interpolate import griddata
from PIL import Image

from lloyd import Field
from Delaunator import Delaunator


class Map:
    """
    Procedurally generated terrain map using Voronoi diagrams.
    
    This class generates terrain by:
    1. Applying Lloyd's relaxation for uniform point distribution
    2. Computing Delaunay triangulation and dual Voronoi diagram
    3. Assigning elevation using Perlin noise with island masking
    4. Assigning moisture for biome classification
    
    Attributes:
        grid_size (float): Map dimensions (default 1.0 for unit square).
        water_level (float): Altitude threshold separating ocean from land.
        noise_scale (float): Perlin noise frequency multiplier.
        cluster (int): Number of island center points.
        points (np.ndarray): Voronoi site coordinates after relaxation.
        triangles (np.ndarray): Delaunay triangulation vertex indices.
        centers (np.ndarray): Voronoi vertices (triangle circumcenters).
        altitudes (np.ndarray): Terrain elevation per region.
        moisture (np.ndarray): Moisture level per region for biome selection.
        polygons (list): Voronoi cell vertices for rendering.
    
    Args:
        p: Initial random points as (N, 2) array.
        size: Map size (default 1.0).
        water_level: Ocean threshold (default 0.4).
        noise_scale: Noise frequency (default 3).
        cluster: Island count (default 4).
    """

    def __init__(self, p, size=1, water_level=0.4, noise_scale=3, cluster=4):
        self.grid_size = size
        self.water_level = water_level
        self.noise_scale = noise_scale
        self.cluster = cluster
        
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
        self.assignMoisture()

    def _build_polygons(self):
        """Compute Voronoi polygon vertices for each region."""
        # Build index: map each point to one of its incoming halfedges
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
            
            # Walk around the Voronoi cell by traversing halfedges
            vertices = []
            e0 = index[i]
            e = e0
            
            while True:
                t = e // 3  # Triangle index from halfedge
                vertices.append(self.centers[t])  # Circumcenter = Voronoi vertex
                
                # Navigate to previous edge in triangle, then jump to opposite
                prev_e = e - 1 if e % 3 != 0 else e + 2
                opp_e = self.halfedges[prev_e]
                
                if opp_e == -1:  # Boundary edge - incomplete polygon
                    break
                
                e = opp_e
                if e == e0:  # Completed the loop
                    break
            
            if len(vertices) > 2:
                self.polygons.append(vertices)
                self.polygon_indices.append(i)

    def assignAltitudes(self):
        """
        Generate terrain elevation using Perlin noise with island masking.
        
        The algorithm combines multi-octave Perlin noise with distance-based
        falloff from randomly placed island centers. A power curve is applied
        to create gentle coastal slopes transitioning to steeper mountain peaks.
        """
        scaled_pts = self.points / self.grid_size * self.noise_scale
        noise_vals = np.array([(pnoise2(x, y, octaves=6) + 1) / 2 
                            for x, y in scaled_pts])
        
        # Randomly place island centers (avoiding edges)
        island_centers = np.random.uniform(0.2, 0.8, (self.cluster, 2)) * self.grid_size
        
        # Distance falloff: points far from island centers become ocean
        all_dists = np.array([np.linalg.norm(self.points - c, axis=1) for c in island_centers])
        distances = np.min(all_dists, axis=0)  # Distance to nearest island center
        normalized_dists = distances / (self.grid_size / 2)
        
        # Combine noise with distance mask (quadratic falloff)
        base_alt = noise_vals - normalized_dists ** 2
        
        # Apply power curve to land for realistic elevation profile
        land_mask = base_alt > self.water_level
        self.altitudes = base_alt.copy()
        land_vals = base_alt[land_mask]
        land_normalized = (land_vals - self.water_level) / (1.0 - self.water_level)
        land_normalized = np.clip(land_normalized, 0, 1)
        
        # Piecewise curve: gentle near coast, steeper inland for mountains
        sharpened = np.where(
            land_normalized < 0.35,
            land_normalized * 0.8,  # Gentle coastal slope
            0.28 + (land_normalized - 0.35) ** 1.3 * 3  # Steeper mountain peaks
        )
        self.altitudes[land_mask] = self.water_level + sharpened * (1.0 - self.water_level)

    def assignMoisture(self):
        """
        Generate moisture values using Perlin noise for biome classification.
        
        Uses a separate noise pattern offset from elevation. Moisture is adjusted
        based on altitude: water regions are fully moist, mid-elevations are drier,
        and high peaks retain moisture for snow accumulation.
        """
        # Use offset coordinates for independent noise pattern
        offset = 500
        scaled_pts = self.points / self.grid_size * self.noise_scale * 1.5
        
        self.moisture = np.array([
            (pnoise2(x + offset, y + offset, octaves=6) + 1) / 2
            for x, y in scaled_pts
        ])
        
        # Adjust moisture based on elevation for realistic biome distribution
        for i in range(len(self.moisture)):
            if self.altitudes[i] < self.water_level:
                self.moisture[i] = 1.0  # Ocean is fully moist
            else:
                land_height = (self.altitudes[i] - self.water_level) / (1.0 - self.water_level)
                if land_height < 0.7:
                    # Rain shadow effect: higher elevation = drier
                    self.moisture[i] *= (1.0 - land_height * 0.25)
                else:
                    # High peaks trap moisture for snow
                    self.moisture[i] = min(1.0, self.moisture[i] * 1.5)

    def plotLand(self, ax):
        """Render Voronoi regions as colored polygons."""
        colors = [self.get_color(self.altitudes[i], self.moisture[i]) for i in self.polygon_indices]
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
        """Calculate circumcenters for all triangles (Voronoi vertices)."""
        tri_indices = self.triangles.reshape(-1, 3)
        
        # Extract triangle vertices
        p1 = self.points[tri_indices[:, 0]]
        p2 = self.points[tri_indices[:, 1]]
        p3 = self.points[tri_indices[:, 2]]
        
        ax, ay = p1[:, 0], p1[:, 1]
        bx, by = p2[:, 0], p2[:, 1]
        cx, cy = p3[:, 0], p3[:, 1]
        
        # Circumcenter formula: intersection of perpendicular bisectors
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        d[d == 0] = 1.0  # Handle collinear points
        
        a_sq = ax**2 + ay**2
        b_sq = bx**2 + by**2
        c_sq = cx**2 + cy**2
        
        ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d
        uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d
        
        return np.column_stack((ux, uy))

    BIOME_COLORS = {
        'OCEAN':        (0.05, 0.30, 0.65),
        'BEACH':        (0.95, 0.90, 0.65),
        'DESERT':       (0.98, 0.85, 0.55),
        'GRASSLAND':    (0.45, 0.85, 0.30),
        'FOREST':       (0.15, 0.55, 0.20),
        'RAINFOREST':   (0.08, 0.45, 0.18),
        'TAIGA':        (0.25, 0.50, 0.35),
        'TUNDRA':       (0.75, 0.70, 0.50),
        'MOUNTAIN':     (0.50, 0.40, 0.30),
        'SNOW':         (0.98, 0.98, 1.00),
    }
    """RGB color palette for each biome type."""

    def get_color(self, alt: float, moisture: float) -> tuple:
        """
        Map altitude and moisture to RGB color.
        
        Args:
            alt: Elevation value.
            moisture: Moisture value (0-1).
            
        Returns:
            RGB tuple (r, g, b) with values in range [0, 1].
        """
        if alt < self.water_level:
            return self.BIOME_COLORS['OCEAN']
        
        max_land = 0.25
        e = (alt - self.water_level) / max_land
        e = np.clip(e, 0, 1)
        m = np.clip(moisture, 0, 1)
        
        if e < 0.1:
            return self.BIOME_COLORS['BEACH']
        if e > 0.85:
            return self.BIOME_COLORS['SNOW']
        if e > 0.60:
            if m < 0.4:
                return self.BIOME_COLORS['MOUNTAIN'] 
            if m < 0.7:
                return self.BIOME_COLORS['TUNDRA']    
            return self.BIOME_COLORS['SNOW']          
        if e > 0.35:
            if m < 0.4:
                return self.BIOME_COLORS['GRASSLAND']  
            return self.BIOME_COLORS['TAIGA']          
        if e > 0.25:
            if m < 0.3:
                return self.BIOME_COLORS['DESERT']    
            if m < 0.6:
                return self.BIOME_COLORS['GRASSLAND']  
            return self.BIOME_COLORS['FOREST']       
        if m < 0.3:
            return self.BIOME_COLORS['DESERT']        
        if m < 0.6:
            return self.BIOME_COLORS['GRASSLAND']      
        return self.BIOME_COLORS['RAINFOREST']        

    def export_heightmap(self, filepath: str, resolution: int = 1024) -> None:
        """
        Export terrain as a grayscale heightmap image.
        
        Creates a PNG image where pixel intensity represents elevation,
        suitable for displacement mapping in 3D applications like Blender.
        
        Args:
            filepath: Output file path.
            resolution: Image dimensions (default 1024x1024).
        """
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        heights = griddata(
            self.points,
            self.altitudes,
            grid_points,
            method="cubic",
            fill_value=0,
        )
        heights = heights.reshape(resolution, resolution)

        h_min, h_max = heights.min(), heights.max()
        heights = (heights - h_min) / (h_max - h_min)
        heights = (heights * 255).astype(np.uint8)
        heights = np.flipud(heights)

        img = Image.fromarray(heights, mode="L")
        img.save(filepath)
        print(f"Heightmap saved: {filepath}")

    def export_colormap(self, filepath: str, resolution: int = 1024) -> None:
        """
        Export terrain biome colors as an RGB image.
        
        Creates a PNG image with biome-based coloring for use as a
        diffuse texture in 3D applications.
        
        Args:
            filepath: Output file path.
            resolution: Image dimensions (default 1024x1024).
        """
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        heights = griddata(
            self.points,
            self.altitudes,
            grid_points,
            method="cubic",
            fill_value=0,
        )
        heights = heights.reshape(resolution, resolution)

        moistures = griddata(
            self.points,
            self.moisture,
            grid_points,
            method="cubic",
            fill_value=0.5,
        )
        moistures = moistures.reshape(resolution, resolution)

        colors = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        for i in range(resolution):
            for j in range(resolution):
                r, g, b = self.get_color(heights[i, j], moistures[i, j])
                colors[i, j] = [int(r * 255), int(g * 255), int(b * 255)]

        colors = np.flipud(colors)

        img = Image.fromarray(colors, mode="RGB")
        img.save(filepath)
        print(f"Colormap saved: {filepath}")


def lloyd(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Apply Lloyd's relaxation algorithm for uniform point distribution.
    
    Args:
        points: Initial point positions as (N, 2) array.
        iterations: Number of relaxation iterations.
        
    Returns:
        Relaxed point positions as (N, 2) array.
    """
    field = Field(points)
    for _ in range(iterations):
        field.relax()
    return field.get_points()