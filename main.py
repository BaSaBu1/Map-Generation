"""
Terrain Visualization Application.

Interactive procedural terrain generator using Voronoi diagrams and Perlin noise.
Provides real-time parameter adjustment through Matplotlib widgets.

Author: Batsambuu Batbold
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from map import Map

# Configuration
NUM_POINTS = 20000
MAP_SIZE = 1.0
DEFAULT_WATER_LEVEL = 0.4
DEFAULT_NOISE_SCALE = 3.0
DEFAULT_CLUSTERS = 5
DEFAULT_HEIGHT = 0.7
DEFAULT_DEPTH = 0.5


class TerrainVisualizer:
    """Manages interactive visualization and UI controls for terrain generation."""

    def __init__(self) -> None:
        """Initialize the terrain visualizer with controls."""
        self.map = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.25)
        self.fig.canvas.manager.set_window_title("Procedural Map Generator")

        self._init_controls()
        self.generate_new_map()

    def _init_controls(self) -> None:
        """Initialize sliders and buttons."""
        ax_noise = plt.axes([0.25, 0.14, 0.65, 0.03])
        ax_height = plt.axes([0.25, 0.09, 0.65, 0.03])
        ax_depth = plt.axes([0.25, 0.04, 0.65, 0.03])
        ax_button = plt.axes([0.8, 0.20, 0.1, 0.04])

        self.slider_noise = Slider(
            ax_noise, 'Noise Scale', 1.0, 10.0, 
            valinit=DEFAULT_NOISE_SCALE, valstep=0.5
        )
        self.slider_height = Slider(
            ax_height, 'Land Height', 0.5, 1.0, 
            valinit=DEFAULT_HEIGHT, valstep=0.05
        )
        self.slider_depth = Slider(
            ax_depth, 'Water Depth', 0.1, 1.0, 
            valinit=DEFAULT_DEPTH, valstep=0.05
        )
        self.btn_new = Button(ax_button, 'New Map')

        self.slider_noise.on_changed(self._on_noise_change)
        self.slider_height.on_changed(self._on_display_change)
        self.slider_depth.on_changed(self._on_display_change)
        self.btn_new.on_clicked(self._on_new_map_clicked)

    def generate_new_map(self) -> None:
        """Generate a new map with fresh random points."""
        points = np.random.rand(NUM_POINTS, 2) * MAP_SIZE
        
        self.map = Map(
            p=points, 
            size=MAP_SIZE, 
            water_level=DEFAULT_WATER_LEVEL,
            noise_scale=self.slider_noise.val, 
            cluster=DEFAULT_CLUSTERS,
            height=self.slider_height.val,
            depth=self.slider_depth.val
        )
        self.refresh_plot()

    def refresh_plot(self) -> None:
        """Redraw the map."""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, MAP_SIZE)
        self.ax.set_ylim(0, MAP_SIZE)
        self.ax.axis('off')

        if self.map:
            self.map.plotLand(self.ax)
        
        self.fig.canvas.draw_idle()

    def _on_noise_change(self, val: float) -> None:
        """Regenerate terrain when noise scale changes."""
        if self.map:
            self.map.noise_scale = val
            self.map.assignAltitudes()
            self.refresh_plot()

    def _on_display_change(self, val: float) -> None:
        """Update colors when height/depth sliders change."""
        if self.map:
            self.map.height = self.slider_height.val
            self.map.depth = self.slider_depth.val
            self.refresh_plot()

    def _on_new_map_clicked(self, event) -> None:
        """Handle new map button click."""
        self.generate_new_map()


if __name__ == "__main__":
    visualizer = TerrainVisualizer()
    plt.show()