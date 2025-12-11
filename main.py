"""
Terrain Visualization Application.

Desktop GUI for interactive procedural terrain generation using Matplotlib.
Provides real-time parameter adjustment through sliders and buttons.

Usage:
    python main.py

Author: Batsambuu Batbold
Course: MATH 437 - Computational Geometry
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

from map import Map

MAP_SIZE = 1.0
DEFAULT_SEED = 7290
NUM_POINTS = 5000
DEFAULT_NOISE_SCALE = 4.0
DEFAULT_WATER_LEVEL = 0.35
DEFAULT_CLUSTERS = 5


class TerrainVisualizer:
    """
    Interactive terrain visualization with Matplotlib GUI controls.
    
    Provides sliders for noise scale, water level, and cluster count,
    plus seed input and random map generation button.
    """

    def __init__(self) -> None:
        """Initialize the visualizer with default settings."""
        self.map = None
        self.current_seed = DEFAULT_SEED

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.30)
        self.fig.canvas.manager.set_window_title("Procedural Map Generator")

        self._init_controls()
        self.generate_new_map()

    def _init_controls(self) -> None:
        """Initialize UI sliders and buttons."""
        ax_seed = plt.axes([0.25, 0.19, 0.15, 0.03])
        self.textbox_seed = TextBox(ax_seed, "Seed ", initial=str(DEFAULT_SEED))
        self.textbox_seed.on_submit(self._on_seed_change)

        ax_noise = plt.axes([0.25, 0.14, 0.65, 0.03])
        ax_water = plt.axes([0.25, 0.09, 0.65, 0.03])
        ax_clusters = plt.axes([0.25, 0.04, 0.65, 0.03])
        ax_button = plt.axes([0.8, 0.19, 0.1, 0.04])

        self.slider_noise = Slider(
            ax_noise, "Noise Scale", 1.0, 10.0,
            valinit=DEFAULT_NOISE_SCALE, valstep=0.5
        )
        self.slider_water = Slider(
            ax_water, "Water Level", 0.0, 0.8,
            valinit=DEFAULT_WATER_LEVEL, valstep=0.05
        )
        self.slider_clusters = Slider(
            ax_clusters, "Clusters", 1, 10,
            valinit=DEFAULT_CLUSTERS, valstep=1
        )
        self.btn_new = Button(ax_button, "New Map")

        self.slider_noise.on_changed(self._on_noise_change)
        self.slider_water.on_changed(self._on_water_change)
        self.slider_clusters.on_changed(self._on_clusters_change)
        self.btn_new.on_clicked(self._on_new_map_clicked)

    def generate_new_map(self) -> None:
        """Generate a new map with the current seed."""
        np.random.seed(self.current_seed)
        points = np.random.rand(NUM_POINTS, 2) * MAP_SIZE

        self.map = Map(
            p=points,
            size=MAP_SIZE,
            water_level=self.slider_water.val,
            noise_scale=self.slider_noise.val,
            cluster=int(self.slider_clusters.val),
        )
        self.refresh_plot()

    def refresh_plot(self) -> None:
        """Redraw the map."""
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, MAP_SIZE)
        self.ax.set_ylim(0, MAP_SIZE)
        self.ax.axis("off")

        if self.map:
            self.map.plotLand(self.ax)

        self.fig.canvas.draw_idle()

    def _on_seed_change(self, text: str) -> None:
        """Update seed and regenerate map."""
        try:
            self.current_seed = int(text)
            self.generate_new_map()
        except ValueError:
            pass

    def _on_noise_change(self, val: float) -> None:
        """Regenerate terrain when noise scale changes."""
        if self.map:
            self.map.noise_scale = val
            self.map.assignAltitudes()
            self.refresh_plot()

    def _on_water_change(self, val: float) -> None:
        """Update water level and recolor."""
        if self.map:
            self.map.water_level = val
            self.refresh_plot()

    def _on_clusters_change(self, val: float) -> None:
        """Regenerate terrain with new cluster count."""
        if self.map:
            self.map.cluster = int(val)
            self.map.assignAltitudes()
            self.refresh_plot()

    def _on_new_map_clicked(self, event) -> None:
        """Generate a new random map."""
        self.current_seed = np.random.randint(0, 10000)
        self.textbox_seed.set_val(str(self.current_seed))
        self.generate_new_map()


if __name__ == "__main__":
    visualizer = TerrainVisualizer()
    plt.show()