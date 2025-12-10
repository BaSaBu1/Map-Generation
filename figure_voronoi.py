"""Quick script to compare random noise vs Perlin noise."""

import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

# Configuration
RESOLUTION = 256
SEED = 42
NOISE_SCALE = 5  # Controls Perlin noise frequency

np.random.seed(SEED)

# Generate random noise (white noise)
random_noise = np.random.rand(RESOLUTION, RESOLUTION)

# Generate Perlin noise
perlin_noise = np.zeros((RESOLUTION, RESOLUTION))
for i in range(RESOLUTION):
    for j in range(RESOLUTION):
        perlin_noise[i, j] = (pnoise2(i / RESOLUTION * NOISE_SCALE, 
                                       j / RESOLUTION * NOISE_SCALE, 
                                       octaves=4) + 1) / 2  # Normalize to 0-1

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Random noise
im1 = axes[0].imshow(random_noise, cmap='terrain', origin='lower')
axes[0].set_title("Random Noise (Static)")
axes[0].set_xticks([])
axes[0].set_yticks([])

# 2. Perlin noise
im2 = axes[1].imshow(perlin_noise, cmap='terrain', origin='lower')
axes[1].set_title("Perlin Noise (Smooth)")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.savefig('report/figures/noise-comparison.png', dpi=150, bbox_inches='tight')
plt.show()
