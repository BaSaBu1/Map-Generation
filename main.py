"""Desktop terrain viewer with Matplotlib sliders for quick parameter tuning.

Run directly to open an interactive window::

    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons

from map import Map

# Defaults shown when the viewer first opens
MAP_SIZE = 1.0
DEFAULT_SEED = 123
DEFAULT_NUM_POINTS = 5000
DEFAULT_NOISE_SCALE = 4.0
DEFAULT_WATER_LEVEL = 0.35
DEFAULT_LAND_CENTERS = 5


class TerrainVisualizer:
    """Matplotlib-based terrain previewer with live slider controls."""

    def __init__(self) -> None:
        """Set up the figure, axes, and UI controls, then render an initial map."""
        self.map: Map | None = None
        self.current_seed = DEFAULT_SEED
        self.num_points = DEFAULT_NUM_POINTS
        self.show_rivers = True
        self.show_biomes = True

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.32)
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("Procedural Map Generator")

        self._init_controls()
        self.generate_new_map()

    def _init_controls(self) -> None:
        """Lay out sliders, seed textbox, point-count textbox, toggles, and New Map button."""
        # Row 1: Seed + Points + New Map button
        ax_seed = plt.axes((0.15, 0.22, 0.12, 0.03))
        self.textbox_seed = TextBox(ax_seed, "Seed ", initial=str(DEFAULT_SEED))
        self.textbox_seed.on_submit(self._on_seed_change)

        ax_pts = plt.axes((0.42, 0.22, 0.12, 0.03))
        self.textbox_points = TextBox(ax_pts, "Points ", initial=str(DEFAULT_NUM_POINTS))
        self.textbox_points.on_submit(self._on_points_change)

        ax_button = plt.axes((0.82, 0.22, 0.10, 0.04))
        self.btn_new = Button(ax_button, "New Map")
        self.btn_new.on_clicked(self._on_new_map_clicked)

        # Row 2-4: Sliders
        ax_noise = plt.axes((0.25, 0.16, 0.65, 0.03))
        ax_water = plt.axes((0.25, 0.11, 0.65, 0.03))
        ax_land_centers = plt.axes((0.25, 0.06, 0.65, 0.03))

        self.slider_noise = Slider(
            ax_noise, "Noise Scale", 1.0, 10.0,
            valinit=DEFAULT_NOISE_SCALE, valstep=0.5
        )
        self.slider_water = Slider(
            ax_water, "Water Level", 0.0, 0.8,
            valinit=DEFAULT_WATER_LEVEL, valstep=0.05
        )
        self.slider_land_centers = Slider(
            ax_land_centers, "Land Centers", 1, 10,
            valinit=DEFAULT_LAND_CENTERS, valstep=1
        )

        # Toggles: Rivers and Biomes
        ax_toggles = plt.axes((0.02, 0.06, 0.12, 0.10))
        self.check_buttons = CheckButtons(
            ax_toggles, ["Rivers", "Biomes"], [True, True]
        )
        self.check_buttons.on_clicked(self._on_toggle_change)

        self.slider_noise.on_changed(self._on_noise_change)
        self.slider_water.on_changed(self._on_water_change)
        self.slider_land_centers.on_changed(self._on_land_centers_change)

    @staticmethod
    def _generate_points(seed: int, num_points: int, map_size: float) -> np.ndarray:
        """Create deterministic random points using a local RNG (avoids global state)."""
        rng = np.random.default_rng(seed)
        return rng.random((num_points, 2)) * map_size

    def _recompute_terrain_fields(self) -> None:
        """Re-derive altitude, moisture, and rivers after a slider change."""
        if self.map is None:
            return
        self.map.assignAltitudes()
        self.map.assignMoisture()
        self.map._build_biome_table()
        self.map.generateRivers()

    def generate_new_map(self) -> None:
        """Build and display a new terrain from the current seed and controls."""
        points = self._generate_points(self.current_seed, self.num_points, MAP_SIZE)

        self.map = Map(
            p=points,
            size=MAP_SIZE,
            water_level=self.slider_water.val,
            noise_scale=self.slider_noise.val,
            land_centers=int(self.slider_land_centers.val),
            seed=self.current_seed,
        )
        self.refresh_plot()

    def refresh_plot(self) -> None:
        """Clear and redraw the terrain on the Matplotlib axes."""
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, MAP_SIZE)
        self.ax.set_ylim(0, MAP_SIZE)
        self.ax.axis("off")

        if self.map:
            if self.show_biomes:
                self.map.plotLand(self.ax)
            else:
                self._plot_land_only()
            if self.show_rivers:
                self.map.plotRivers(self.ax)

        self.fig.canvas.draw_idle()

    def _plot_land_only(self) -> None:
        """Draw land/ocean with flat colors (no biome coloring)."""
        from matplotlib.collections import PolyCollection

        assert self.map is not None
        colors = []
        for i in self.map.polygon_indices:
            alt = self.map.altitudes[i]
            if alt < self.map.water_level:
                colors.append((0.12, 0.40, 0.70))
            else:
                colors.append((0.45, 0.72, 0.30))
        pc = PolyCollection(self.map.polygons, facecolors=colors, edgecolors='none')
        self.ax.add_collection(pc)

    def _on_seed_change(self, text: str) -> None:
        """Parse a new seed from the textbox and regenerate the map."""
        try:
            self.current_seed = int(text)
            self.generate_new_map()
        except ValueError:
            self.textbox_seed.set_val(str(self.current_seed))  # revert on bad input

    def _on_points_change(self, text: str) -> None:
        """Parse new point count from the textbox and regenerate the map."""
        try:
            val = int(text)
            if val < 50:
                val = 50
            elif val > 100000:
                val = 100000
            self.num_points = val
            self.generate_new_map()
        except ValueError:
            self.textbox_points.set_val(str(self.num_points))

    def _on_toggle_change(self, label: str | None) -> None:
        """Toggle rivers or biomes display and redraw."""
        if label == "Rivers":
            self.show_rivers = not self.show_rivers
        elif label == "Biomes":
            self.show_biomes = not self.show_biomes
        self.refresh_plot()

    def _on_noise_change(self, val: float) -> None:
        """Update noise scale and recompute terrain."""
        if self.map:
            self.map.noise_scale = val
            self._recompute_terrain_fields()
            self.refresh_plot()

    def _on_water_change(self, val: float) -> None:
        """Update water level and rebuild rivers."""
        if self.map:
            self.map.water_level = val
            self._recompute_terrain_fields()
            self.refresh_plot()

    def _on_land_centers_change(self, val: float) -> None:
        """Update land center count and recompute terrain."""
        if self.map:
            self.map.land_centers = int(val)
            self._recompute_terrain_fields()
            self.refresh_plot()

    def _on_new_map_clicked(self, event) -> None:
        """Pick a random seed and regenerate."""
        self.current_seed = int(np.random.default_rng().integers(0, 10000))
        self.textbox_seed.set_val(str(self.current_seed))
        self.generate_new_map()


if __name__ == "__main__":
    visualizer = TerrainVisualizer()
    plt.show()