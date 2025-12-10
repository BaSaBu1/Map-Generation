"""
Procedural Terrain Map Generation Module.

Generates 2D Voronoi-based terrain maps using Lloyd's relaxation and Perlin noise.

Author: Batsambuu Batbold
Date: December 2025
"""

# TODO: Rivers, Trees

import sys
import os

# Add external libraries to path BEFORE importing
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
        # Vectorized noise
        scaled_pts = self.points / self.grid_size * self.noise_scale
        noise_vals = np.array([(pnoise2(x, y, octaves=6) + 1) / 2 
                            for x, y in scaled_pts])
        
        # Generate island centers
        island_centers = np.random.uniform(0.2, 0.8, (self.cluster, 2)) * self.grid_size
        
        # Vectorized distance to nearest island center
        all_dists = np.array([np.linalg.norm(self.points - c, axis=1) for c in island_centers])
        distances = np.min(all_dists, axis=0)
        normalized_dists = distances / (self.grid_size / 2)
        
        # Base altitude
        base_alt = noise_vals - normalized_dists ** 2
        
        # Power curve
        land_mask = base_alt > self.water_level
        self.altitudes = base_alt.copy()
        land_vals = base_alt[land_mask]
        land_normalized = (land_vals - self.water_level) / (1.0 - self.water_level)
        land_normalized = np.clip(land_normalized, 0, 1)
        sharpened = np.where(
            land_normalized < 0.3, # % of low altitude land
            land_normalized * 0.5, # gentle slope near water
            0.15 + (land_normalized - 0.3) ** 1.7 * 4 # steeper inland, the power controls steepness, multiplier controls max height
        )
        self.altitudes[land_mask] = self.water_level + sharpened * (1.0 - self.water_level)

    def assignMoisture(self):
        """Generate moisture using Perlin noise with different frequency/offset."""
        # Use different noise coordinates to get independent moisture pattern
        offset = 500  # Offset to get different noise pattern
        scaled_pts = self.points / self.grid_size * self.noise_scale * 1.5
        
        self.moisture = np.array([
            (pnoise2(x + offset, y + offset, octaves=4) + 1) / 2
            for x, y in scaled_pts
        ])
        
        # Adjust moisture based on altitude
        for i in range(len(self.moisture)):
            if self.altitudes[i] < self.water_level:
                self.moisture[i] = 1.0  # Water is fully "moist"
            else:
                # Normalize land altitude
                land_height = (self.altitudes[i] - self.water_level) / (1.0 - self.water_level)
                
                # Mid-elevation mountains are drier, but high peaks can trap snow
                if land_height < 0.7:
                    # Reduce moisture for mid-elevations
                    self.moisture[i] *= (1.0 - land_height * 0.25)
                else:
                    # High peaks: boost moisture for snow accumulation
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

    # Biome color palette - Sharp, artistic, vibrant colors
    BIOME_COLORS = {
        'OCEAN':        (0.05, 0.30, 0.65),  # Vivid deep blue
        'BEACH':        (0.95, 0.90, 0.65),  # Bright golden sand
        'DESERT':       (0.98, 0.85, 0.55),  # Warm saturated tan
        'GRASSLAND':    (0.45, 0.85, 0.30),  # Vibrant lime green
        'FOREST':       (0.15, 0.55, 0.20),  # Rich emerald green
        'RAINFOREST':   (0.08, 0.45, 0.18),  # Deep jungle green
        'TAIGA':        (0.25, 0.50, 0.35),  # Cool forest green
        'TUNDRA':       (0.75, 0.70, 0.50),  # Warm earthy tan
        'MOUNTAIN':     (0.50, 0.40, 0.30),  # Rich brown stone
        'SNOW':         (0.98, 0.98, 1.00),  # Brilliant white
    }

    def get_color(self, alt: float, moisture: float) -> tuple:
        """
        Map altitude and moisture to RGB color based on 10 distinct biomes.
        
        Elevation zones (low → high):
            - Low: Ocean, Beach, Desert/Grassland/Rainforest (tropical)
            - Mid: Desert/Grassland/Forest (temperate)
            - Upper-mid: Taiga (boreal forest)
            - High: Tundra, Mountain, Snow
        """
        # Normalize altitude: 0 = water level, 1 = max land
        if alt < self.water_level:
            return self.BIOME_COLORS['OCEAN']
        
        # Normalize land elevation (0 to 1)
        max_land = 0.25
        e = (alt - self.water_level) / max_land
        e = np.clip(e, 0, 1)
        m = np.clip(moisture, 0, 1)
        
        # Beach zone (very low elevation)
        if e < 0.08:
            return self.BIOME_COLORS['BEACH']
        
        # Highest peaks always snow (guarantee snow on tallest mountains)
        if e > 0.75:
            return self.BIOME_COLORS['SNOW']
        
        # High elevation (mountains)
        if e > 0.50:
            if m < 0.4:
                return self.BIOME_COLORS['MOUNTAIN'] 
            if m < 0.7:
                return self.BIOME_COLORS['TUNDRA']    
            return self.BIOME_COLORS['SNOW']          
        
        # Upper-mid elevation (boreal)
        if e > 0.35:
            if m < 0.4:
                return self.BIOME_COLORS['GRASSLAND']  
            return self.BIOME_COLORS['TAIGA']          
        
        # Mid elevation (temperate)
        if e > 0.25:
            if m < 0.3:
                return self.BIOME_COLORS['DESERT']    
            if m < 0.6:
                return self.BIOME_COLORS['GRASSLAND']  
            return self.BIOME_COLORS['FOREST']       
        
        # Low elevation (tropical)
        if m < 0.3:
            return self.BIOME_COLORS['DESERT']        
        if m < 0.6:
            return self.BIOME_COLORS['GRASSLAND']      
        return self.BIOME_COLORS['RAINFOREST']        

    def export_heightmap(self, filepath: str, resolution: int = 1024) -> None:
        """Export terrain as a grayscale heightmap image for Blender."""

        # Create regular grid
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        # Interpolate altitudes onto grid
        heights = griddata(
            self.points,
            self.altitudes,
            grid_points,
            method="cubic",
            fill_value=0,
        )
        heights = heights.reshape(resolution, resolution)

        # Normalize to 0-255
        h_min, h_max = heights.min(), heights.max()
        heights = (heights - h_min) / (h_max - h_min)
        heights = (heights * 255).astype(np.uint8)

        # Flip vertically for correct Blender orientation
        heights = np.flipud(heights)

        img = Image.fromarray(heights, mode="L")
        img.save(filepath)
        print(f"Heightmap saved: {filepath}")

    def export_colormap(self, filepath: str, resolution: int = 1024) -> None:
        """Export terrain colors as an RGB image for Blender texturing."""

        # Create regular grid
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        # Interpolate altitudes onto grid
        heights = griddata(
            self.points,
            self.altitudes,
            grid_points,
            method="cubic",
            fill_value=0,
        )
        heights = heights.reshape(resolution, resolution)

        # Interpolate moisture onto grid
        moistures = griddata(
            self.points,
            self.moisture,
            grid_points,
            method="cubic",
            fill_value=0.5,
        )
        moistures = moistures.reshape(resolution, resolution)

        # Generate colors from altitudes and moisture
        colors = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        for i in range(resolution):
            for j in range(resolution):
                r, g, b = self.get_color(heights[i, j], moistures[i, j])
                colors[i, j] = [int(r * 255), int(g * 255), int(b * 255)]

        # Flip vertically for correct Blender orientation
        colors = np.flipud(colors)

        img = Image.fromarray(colors, mode="RGB")
        img.save(filepath)
        print(f"Colormap saved: {filepath}")


def lloyd(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply Lloyd's relaxation for uniform point distribution."""
    field = Field(points)
    for _ in range(iterations):
        field.relax()
    return field.get_points()