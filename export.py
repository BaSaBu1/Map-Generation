"""
Blender Export Script.

Generates heightmap and colormap images for 3D terrain visualization in Blender.
Output files are saved to the outputs/ directory.

Usage:
    python export.py

Author: Batsambuu Batbold
Course: MATH 437 - Computational Geometry
Date: December 2025
"""

import os
import time

import numpy as np

from map import Map

SEED = 437
NUM_POINTS = 200000
NOISE_SCALE = 4.0
WATER_LEVEL = 0.35
CLUSTERS = 5
RESOLUTION = 2048

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
OUTPUT_HEIGHTMAP = os.path.join(OUTPUT_DIR, "heightmap.png")
OUTPUT_COLORMAP = os.path.join(OUTPUT_DIR, "colormap.png")


def main() -> None:
    """Generate and export terrain data for Blender."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating terrain...")
    np.random.seed(SEED)
    points = np.random.rand(NUM_POINTS, 2)
    
    terrain = Map(
        points,
        size=1,
        water_level=WATER_LEVEL,
        noise_scale=NOISE_SCALE,
        cluster=CLUSTERS,
    )

    print(f"Exporting at {RESOLUTION}x{RESOLUTION} resolution...")
    start_time = time.time()
    terrain.export_heightmap(OUTPUT_HEIGHTMAP, resolution=RESOLUTION)
    terrain.export_colormap(OUTPUT_COLORMAP, resolution=RESOLUTION)
    print(f"Export complete in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
