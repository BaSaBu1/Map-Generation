"""
Blender Export Script.

Exports terrain data as heightmap and colormap images for 3D visualization in Blender.

Author: Batsambuu Batbold
Date: December 2025
"""

import time
import os
import numpy as np
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from map import Map

# Configuration
SEED = 44536
NUM_POINTS = 100000
NOISE_SCALE = 4.0
WATER_LEVEL = 0.35
CLUSTERS = 5
RESOLUTION = 1024

# Output paths
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
OUTPUT_HEIGHTMAP = os.path.join(OUTPUT_DIR, "heightmap.png")
OUTPUT_COLORMAP = os.path.join(OUTPUT_DIR, "colormap.png")


def main() -> None:
    """Generate and export terrain data for Blender."""
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
    elapsed = time.time() - start_time
    print(f"Export complete. Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
