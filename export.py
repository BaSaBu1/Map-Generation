"""CLI exporter for terrain textures (heightmap, colormap, river mask).

Given the same seed and parameters, this script always produces identical outputs,
making it suitable for reproducible asset pipelines::

    python export.py --seed 123 --points 50000 --resolution 4096
"""

import argparse
import os
import time

import numpy as np

from map import Map

# Default values (overridden by CLI flags)
SEED = 123
NUM_POINTS = 50000
NOISE_SCALE = 4.0
WATER_LEVEL = 0.35
LAND_CENTERS = 5
RESOLUTION = 4096

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments for the export run."""
    parser = argparse.ArgumentParser(
        description="Export terrain textures for 3D rendering.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--points", type=int, default=NUM_POINTS,
                        help="Number of Voronoi points")
    parser.add_argument("--noise-scale", type=float, default=NOISE_SCALE,
                        help="Perlin noise scale")
    parser.add_argument("--water-level", type=float, default=WATER_LEVEL,
                        help="Water level threshold")
    parser.add_argument("--land-centers", type=int, default=LAND_CENTERS,
                        help="Number of terrain land anchor points")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="Output texture resolution (square)")
    parser.add_argument("--river-width-scale", type=float, default=0.90,
                        help="Scale factor for river thickness in exported maps")
    parser.add_argument("--river-opacity", type=float, default=0.72,
                        help="River blend strength in exported colormap (0-1)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory where output textures are written")
    return parser.parse_args()


def main() -> None:
    """Build one terrain and write all three texture files."""
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    output_files = {
        "heightmap": os.path.join(output_dir, "heightmap.png"),
        "colormap": os.path.join(output_dir, "colormap.png"),
        "rivermap": os.path.join(output_dir, "rivermap.png"),
    }

    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating terrain...")
    rng = np.random.default_rng(args.seed)
    points = rng.random((args.points, 2))
    
    terrain = Map(
        points,
        size=1,
        water_level=args.water_level,
        noise_scale=args.noise_scale,
        land_centers=args.land_centers,
    )

    print(f"Exporting at {args.resolution}x{args.resolution} resolution...")
    start_time = time.time()

    print("  [1/3] Heightmap (16-bit grayscale)...")
    terrain.export_heightmap(output_files["heightmap"], resolution=args.resolution)

    print("  [2/3] Colormap (biome texture + rivers)...")
    terrain.export_colormap(
        output_files["colormap"],
        resolution=args.resolution,
        river_opacity=args.river_opacity,
        river_width_scale=args.river_width_scale,
    )

    print("  [3/3] River mask (grayscale, for shader mixing)...")
    terrain.export_rivermap(
        output_files["rivermap"],
        resolution=args.resolution,
        river_width_scale=args.river_width_scale,
    )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s  →  {output_dir}")
    print("\nFiles:")
    print("  heightmap.png  - 16-bit grayscale elevation")
    print("  colormap.png   - RGB biome texture with rivers")
    print("  rivermap.png   - grayscale river mask")


if __name__ == "__main__":
    main()
