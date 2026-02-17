"""Core terrain engine shared by all frontends (desktop, Streamlit, CLI exporter).

Builds a Voronoi mesh from relaxed random points, assigns elevation and
moisture via Perlin noise, classifies biomes, routes drainage to the ocean,
and exports textures for 3D rendering.
"""

import sys
import os
import heapq

sys.path.append(os.path.join(os.path.dirname(__file__), "Delaunator-Python"))
sys.path.append(os.path.join(os.path.dirname(__file__), "lloyd"))

import numpy as np
from matplotlib.collections import PolyCollection
from noise import pnoise2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

from lloyd.lloyd import Field
from Delaunator import Delaunator


class Map:
    """Procedural terrain built on a Voronoi/Delaunay dual mesh.

    Construction pipeline:
        1. Lloyd-relax input points for even spacing.
        2. Triangulate (Delaunay) and compute circumcenters (Voronoi vertices).
        3. Assign elevation and moisture with Perlin noise.
        4. Build a blurred biome color table.
        5. Route drainage and extract river segments.
    """

    def __init__(self, p: np.ndarray, size: float = 1.0, water_level: float = 0.4,
                 noise_scale: float = 3.0, land_centers: int = 4,
                 custom_anchors: list[tuple[float, float]] | None = None,
                 seed: int | None = None):
        """Create a terrain from random or user-placed anchor points.

        Args:
            p:              (N, 2) array of initial site coordinates.
            size:           World extent used for scaling and plotting.
            water_level:    Elevation cutoff between water and land.
            noise_scale:    Perlin noise frequency multiplier.
            land_centers:   Number of random anchor points (ignored when
                            *custom_anchors* is provided).
            custom_anchors: Optional list of (x, y) anchor positions in
                            [0, 1] normalized coordinates.  When given,
                            these replace random anchor placement.
            seed:           Random seed for reproducible anchor placement.
                            When *None*, a fresh seed is chosen once and
                            stored so that subsequent calls (e.g.
                            ``assignAltitudes``) remain deterministic.
        """
        self.grid_size = size
        self.water_level = water_level
        self.noise_scale = noise_scale
        self.land_centers = land_centers
        self.custom_anchors = custom_anchors
        self._rng = np.random.default_rng(seed)

        # Relax points for uniform spacing
        self.points = lloyd(p, 3)
        self.numRegions = len(self.points)

        # Triangulate and derive the Voronoi dual
        delaunay = Delaunator(self.points)
        self.triangles = np.array(delaunay.triangles)
        self.halfedges = np.array(delaunay.halfedges)
        self.numTriangles = len(self.triangles) // 3
        self.centers = self._get_circumcenters()  # Voronoi vertices

        self._build_polygons()
        self.assignAltitudes()
        self.assignMoisture()
        self._build_biome_table()
        self.generateRivers()

    def _build_polygons(self) -> None:
        """Collect Voronoi cell polygons for rendering."""
        # Map each site to one of its incoming halfedges
        index = np.full(self.numRegions, -1, dtype=int)
        for e in range(len(self.halfedges)):
            if index[self.triangles[e]] == -1:
                index[self.triangles[e]] = e

        self.polygons = []
        self.polygon_indices = []

        for i in range(self.numRegions):
            if index[i] == -1:
                continue

            # Walk the halfedge ring to collect circumcenters around this cell
            vertices = []
            e0 = index[i]
            e = e0
            while True:
                vertices.append(self.centers[e // 3])
                prev_e = e - 1 if e % 3 != 0 else e + 2
                opp_e = self.halfedges[prev_e]
                if opp_e == -1:       # boundary - cell is open
                    break
                e = opp_e
                if e == e0:           # back to start - cell complete
                    break

            if len(vertices) > 2:
                self.polygons.append(vertices)
                self.polygon_indices.append(i)

    def assignAltitudes(self) -> None:
        """Set per-region elevation using Perlin noise + land-anchor falloff.

        When *custom_anchors* is set, those positions are used directly;
        otherwise random interior points are generated.  Regions far from
        every anchor sink below ``water_level`` and become ocean.  A piecewise
        power curve flattens coastlines and steepens mountains.
        """
        scaled_pts = self.points / self.grid_size * self.noise_scale
        noise_vals = np.array([(pnoise2(x, y, octaves=6) + 1) / 2
                               for x, y in scaled_pts])

        # Use user-provided anchors, or place random ones away from edges
        if self.custom_anchors is not None:
            anchors = np.array(self.custom_anchors) * self.grid_size
        else:
            anchors = self._rng.uniform(0.2, 0.8, (self.land_centers, 2)) * self.grid_size

        # Quadratic distance falloff from the nearest anchor
        all_dists = np.array([np.linalg.norm(self.points - c, axis=1) for c in anchors])
        normalized_dists = np.min(all_dists, axis=0) / (self.grid_size / 2)
        base_alt = noise_vals - normalized_dists ** 2

        # Shape land elevation with a two-segment power curve
        land_mask = base_alt > self.water_level
        self.altitudes = base_alt.copy()
        land_vals = base_alt[land_mask]
        land_norm = np.clip((land_vals - self.water_level) / (1.0 - self.water_level), 0, 1)

        coastal = land_norm < 0.35
        sharpened = np.empty_like(land_norm)
        sharpened[coastal] = land_norm[coastal] * 0.8            # gentle coast
        sharpened[~coastal] = (
            0.28 + np.power(land_norm[~coastal] - 0.35, 1.3) * 3  # steep highlands
        )
        self.altitudes[land_mask] = self.water_level + sharpened * (1.0 - self.water_level)

    def assignMoisture(self) -> None:
        """Set per-region moisture from an independent Perlin noise field.

        Moisture is adjusted by elevation: ocean is fully wet, mid-altitude land
        gets a slight drying effect, and high peaks retain moisture (snow).
        """
        offset = 500  # shift so moisture noise is uncorrelated with elevation
        scaled_pts = self.points / self.grid_size * self.noise_scale * 1.5

        self.moisture = np.array([
            (pnoise2(x + offset, y + offset, octaves=6) + 1) / 2
            for x, y in scaled_pts
        ])

        for i in range(len(self.moisture)):
            if self.altitudes[i] < self.water_level:
                self.moisture[i] = 1.0
            else:
                h = (self.altitudes[i] - self.water_level) / (1.0 - self.water_level)
                if h < 0.7:
                    self.moisture[i] *= (1.0 - h * 0.25)   # drier at higher land
                else:
                    self.moisture[i] = min(1.0, self.moisture[i] * 1.5)  # snow peaks

    def _build_biome_table(self) -> None:
        """Pre-compute a Gaussian-blurred biome color LUT indexed by (elevation, moisture)."""
        res = self.BIOME_TABLE_RES
        table = np.zeros((res, res, 3))

        for i in range(res):
            for j in range(res):
                table[i, j] = self._get_raw_biome_color(i / (res - 1), j / (res - 1))

        for c in range(3):
            table[:, :, c] = gaussian_filter(table[:, :, c], sigma=8.0, mode='nearest')

        self._biome_table = table

    def _build_vertex_adjacency(self, num_v: int) -> list[list[int]]:
        """Return adjacency lists between Voronoi vertices (triangle indices)."""
        adj = [[] for _ in range(num_v)]
        edge_set: set[tuple[int, int]] = set()
        for e in range(len(self.halfedges)):
            opp = self.halfedges[e]
            if opp == -1:
                continue
            t1, t2 = e // 3, opp // 3
            if t1 == t2 or (t1, t2) in edge_set:
                continue
            adj[t1].append(t2)
            adj[t2].append(t1)
            edge_set.add((t1, t2))
            edge_set.add((t2, t1))
        return adj

    def _route_flow_to_water(
        self,
        adj: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Route every land vertex to a water outlet via priority-flood.

        Returns:
            flow_to:    Next vertex index on the path toward water (-1 = none).
            spill:      Minimum spill elevation (max altitude along path to ocean).
            water_mask: Boolean mask of vertices below water level.
        """
        num_v = len(adj)
        water_mask = self.vertex_alt < self.water_level

        spill = np.full(num_v, np.inf, dtype=float)
        flow_to = np.full(num_v, -1, dtype=int)
        pq: list[tuple[float, int]] = []

        water_vertices = np.where(water_mask)[0]
        if water_vertices.size == 0:
            # No water at all - pick the lowest vertex as an artificial outlet
            outlet = int(np.argmin(self.vertex_alt))
            water_vertices = np.array([outlet], dtype=int)
            water_mask[outlet] = True

        for v in water_vertices:
            spill[v] = float(self.vertex_alt[v])
            heapq.heappush(pq, (float(spill[v]), int(v)))

        # Flood outward from water, tracking the lowest-spill path
        while pq:
            curr_spill, v = heapq.heappop(pq)
            if curr_spill > spill[v]:
                continue
            for n in adj[v]:
                cand_spill = max(curr_spill, self.vertex_alt[n])
                if cand_spill < spill[n]:
                    spill[n] = cand_spill
                    flow_to[n] = v
                    heapq.heappush(pq, (cand_spill, n))

        # Prefer a steeper downhill neighbor when it still drains correctly
        for v in range(num_v):
            if water_mask[v]:
                continue
            best, best_drop = flow_to[v], 0.0
            for n in adj[v]:
                if spill[n] > spill[v] + 1e-12:
                    continue
                drop = self.vertex_alt[v] - self.vertex_alt[n]
                if drop > best_drop:
                    best_drop, best = drop, n
            if best != -1:
                flow_to[v] = best

        return flow_to, spill, water_mask

    def _trace_path_to_water(
        self, source: int, flow_to: np.ndarray, water_mask: np.ndarray,
    ) -> tuple[list[int], bool]:
        """Follow ``flow_to`` links from *source* until water is reached.

        Returns the list of vertices visited and whether water was found.
        """
        path: list[int] = []
        visited: set[int] = set()
        v = source

        while True:
            if v in visited:
                return path, False
            visited.add(v)

            if water_mask[v]:
                return path, True

            nxt = int(flow_to[v])
            if nxt == -1:
                return path, False

            path.append(v)
            v = nxt

    def generateRivers(self) -> None:
        """Build the river network with a two-pass source selection.

        Pass 1 picks long, high-flow main channels.  Pass 2 adds shorter
        tributaries that feed into the existing network, filling out the
        drainage so that large maps still look well-covered.
        """
        num_v = self.numTriangles

        # Average altitude of each triangle's corners = Voronoi vertex elevation
        tri_indices = self.triangles.reshape(-1, 3)
        self.vertex_alt = np.mean(self.altitudes[tri_indices], axis=1)

        adj = self._build_vertex_adjacency(num_v)
        flow_to, spill, water_mask = self._route_flow_to_water(adj)

        # Accumulate flow: process from highest spill down so totals propagate
        self.flow_acc = np.ones(num_v, dtype=float)
        order = np.argsort(-spill)
        for v in order:
            if flow_to[v] != -1:
                self.flow_acc[flow_to[v]] += self.flow_acc[v]

        self.flow_to = flow_to
        self.max_flow = float(np.max(self.flow_acc)) if num_v > 0 else 1.0

        land_mask = ~water_mask
        if not np.any(land_mask):
            self.rivers = []
            return

        # Count how many hops each vertex is from water
        path_len = self._compute_path_lengths(num_v, land_mask, water_mask, flow_to)

        # Pass 1 - main rivers (long, high-flow)
        main_min_len = max(5, int(np.sqrt(num_v) * 0.05))
        main_paths = self._select_rivers(
            land_mask, water_mask, flow_to, path_len,
            min_path_len=main_min_len,
            top_fraction=0.07,
            max_count_factor=0.45,
            occupancy_spacing=4,
        )

        # Pass 2 - tributaries (shorter, moderate-flow)
        trib_min_len = max(3, main_min_len // 2)
        trib_paths = self._select_rivers(
            land_mask, water_mask, flow_to, path_len,
            min_path_len=trib_min_len,
            top_fraction=0.15,
            max_count_factor=0.40,
            occupancy_spacing=3,
            existing_paths=main_paths,
        )

        all_paths = main_paths + trib_paths

        if not all_paths:
            self.rivers = []
            return

        self.rivers = self._paths_to_segments(all_paths, flow_to, water_mask)
        if self.rivers:
            self.max_flow = float(max(flow for _, _, flow in self.rivers))

    # ------------------------------------------------------------------
    # River helper methods
    # ------------------------------------------------------------------

    def _compute_path_lengths(
        self, num_v: int, land_mask: np.ndarray,
        water_mask: np.ndarray, flow_to: np.ndarray,
    ) -> np.ndarray:
        """Return the number of flow hops from each vertex to water (-1 if unreachable)."""
        path_len = np.full(num_v, -1, dtype=int)
        path_len[water_mask] = 0
        for v in np.where(land_mask)[0]:
            if path_len[v] >= 0:
                continue
            path, reached = self._trace_path_to_water(int(v), flow_to, water_mask)
            base = 0 if reached else -1
            for node in reversed(path):
                if base < 0:
                    path_len[node] = -1
                else:
                    base += 1
                    path_len[node] = base
        return path_len

    def _select_rivers(
        self,
        land_mask: np.ndarray,
        water_mask: np.ndarray,
        flow_to: np.ndarray,
        path_len: np.ndarray,
        *,
        min_path_len: int,
        top_fraction: float,
        max_count_factor: float,
        occupancy_spacing: int,
        existing_paths: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """Score candidate source vertices and trace the best ones to water.

        Args:
            min_path_len:      Reject rivers shorter than this (hops).
            top_fraction:      Keep sources in the top *this* fraction of flow.
            max_count_factor:  Multiplied by sqrt(num_v) to cap the river count.
            occupancy_spacing: Spacing for marking nodes "taken" (lower = stricter).
            existing_paths:    Already-selected paths to avoid overlapping.
        """
        num_v = len(land_mask)
        long_sources = land_mask & (path_len >= min_path_len)
        if not np.any(long_sources):
            long_sources = land_mask & (path_len > 0)

        source_flows = self.flow_acc[long_sources]
        if source_flows.size == 0:
            return []

        # Only keep sources above a flow-accumulation threshold
        quantile = np.clip(1.0 - top_fraction, 0.0, 0.999)
        quantile_min = float(np.quantile(source_flows, quantile))
        fallback_min = float(max(2.0, num_v * 0.0006))
        min_source_flow = max(fallback_min, quantile_min)

        candidates = np.where(long_sources & (self.flow_acc >= min_source_flow))[0]
        if candidates.size == 0:
            candidates = np.where(long_sources)[0]

        # Score = flow * relative path length - favors long, strong rivers
        max_len = max(int(np.max(path_len[long_sources])), 1)
        scores = self.flow_acc[candidates] * (1.0 + path_len[candidates] / max_len)
        ordered = candidates[np.argsort(-scores)]

        max_rivers = max(15, int(np.sqrt(num_v) * max_count_factor))

        # Seed occupancy with previously chosen paths to avoid overlap
        occupied = np.zeros(num_v, dtype=bool)
        if existing_paths:
            for p in existing_paths:
                for idx in range(0, len(p), max(1, occupancy_spacing)):
                    occupied[p[idx]] = True

        selected: list[list[int]] = []
        for src in ordered:
            if len(selected) >= max_rivers:
                break
            if occupied[src]:
                continue

            path, reached = self._trace_path_to_water(int(src), flow_to, water_mask)
            if not reached or len(path) < min_path_len:
                continue

            selected.append(path)
            step = max(1, occupancy_spacing)
            for idx in range(0, len(path), step):
                occupied[path[idx]] = True

        return selected

    def _paths_to_segments(
        self, paths: list[list[int]], flow_to: np.ndarray, water_mask: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Convert vertex-index paths into (start, end, flow) line segments for drawing."""
        segments: list[tuple[np.ndarray, np.ndarray, float]] = []
        seen: set[tuple[int, int]] = set()
        margin = self.grid_size * 0.02

        for path in paths:
            for v in path:
                target = int(flow_to[v])
                if target == -1 or self.vertex_alt[v] < self.water_level:
                    continue
                seg_key = (v, target)
                if seg_key in seen:
                    continue

                p1, p2 = self.centers[v], self.centers[target]
                if not (
                    -margin <= p1[0] <= self.grid_size + margin
                    and -margin <= p1[1] <= self.grid_size + margin
                    and -margin <= p2[0] <= self.grid_size + margin
                    and -margin <= p2[1] <= self.grid_size + margin
                ):
                    continue

                seen.add(seg_key)
                segments.append((p1, p2, self.flow_acc[v]))

        return segments

    def plotLand(self, ax) -> None:
        """Draw biome-colored Voronoi cells onto *ax*."""
        colors = [self.get_color(float(self.altitudes[i]), float(self.moisture[i]))
                  for i in self.polygon_indices]
        pc = PolyCollection(self.polygons, facecolors=colors, edgecolors='none')
        ax.add_collection(pc)

    def plotRivers(self, ax) -> None:
        """Draw rivers with width and opacity scaled by flow accumulation."""
        if not self.rivers:
            return

        # Keep rivers proportional to Voronoi cell size across point counts.
        # As numRegions increases, cells get smaller, so river strokes should thin out.
        density_scale = np.clip(
            np.sqrt(2000.0 / max(float(self.numRegions), 1.0)),
            0.22,
            1.0,
        )

        for start, end, flow in self.rivers:
            t = (flow / self.max_flow) ** 0.45
            width = max(0.08, (0.22 + 1.80 * t) * density_scale)
            blue = 0.85 - 0.25 * t
            color = (0.05, 0.15 + 0.15 * t, blue)
            alpha = min(0.90, 0.35 + 0.45 * t)
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=color, linewidth=width, alpha=alpha,
                    solid_capstyle='round', zorder=2)

    def plotVoronoi(self, ax) -> None:
        """Overlay Voronoi cell edges (debug/visualization helper)."""
        for e in range(len(self.halfedges)):
            if self.halfedges[e] != -1 and e < self.halfedges[e]:
                t1 = e // 3
                t2 = self.halfedges[e] // 3
                ax.plot([self.centers[t1][0], self.centers[t2][0]], 
                        [self.centers[t1][1], self.centers[t2][1]], 
                        'b-', linewidth=1, color='gray', alpha=0.5)

    def plotDelaunay(self, ax) -> None:
        """Overlay Delaunay triangulation edges (debug/visualization helper)."""
        ax.triplot(self.points[:, 0], self.points[:, 1], self.triangles, 
                'b-', linewidth=0.5, alpha=0.5)

    def _get_circumcenters(self) -> np.ndarray:
        """Vectorized circumcenter calculation for every triangle."""
        tri_indices = self.triangles.reshape(-1, 3)
        p1 = self.points[tri_indices[:, 0]]
        p2 = self.points[tri_indices[:, 1]]
        p3 = self.points[tri_indices[:, 2]]
        
        ax, ay = p1[:, 0], p1[:, 1]
        bx, by = p2[:, 0], p2[:, 1]
        cx, cy = p3[:, 0], p3[:, 1]

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        d[d == 0] = 1.0  # degenerate (collinear) guard
        
        a_sq = ax**2 + ay**2
        b_sq = bx**2 + by**2
        c_sq = cx**2 + cy**2
        
        ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d
        uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d
        
        return np.column_stack((ux, uy))

    BIOME_TABLE_RES = 256
    """Side length of the (elevation x moisture) biome color LUT."""

    COLOR_GRADE = {
        "saturation": 1.14,
        "contrast": 1.06,
        "gamma": 0.96,
        "vibrance": 0.10,
    }
    """Color grading applied to exported colormaps for richer albedo."""

    BIOME_COLORS = {
        'DEEP_OCEAN':   (0.02, 0.15, 0.50),
        'SHALLOW_OCEAN':(0.12, 0.40, 0.70),
        'BEACH':        (0.95, 0.90, 0.65),
        'DESERT':       (0.98, 0.85, 0.55),
        'GRASSLAND':    (0.45, 0.85, 0.30),
        'FOREST':       (0.15, 0.55, 0.20),
        'RAINFOREST':   (0.08, 0.45, 0.18),
        'TAIGA':        (0.25, 0.50, 0.35),
        'TUNDRA':       (0.75, 0.70, 0.50),
        'MOUNTAIN':     (0.50, 0.40, 0.30),
        'SNOW':         (0.98, 0.98, 1.00),
        'RIVER':        (0.05, 0.25, 0.65),
    }
    """RGB palette for each biome (values in 0-1)."""

    def _get_raw_biome_color(self, e: float, m: float) -> tuple[float, float, float]:
        """Whittaker-style hard-boundary biome lookup (blurred later by the LUT builder).

        Args:
            e: Normalized elevation (0 = coast, 1 = peak).
            m: Moisture (0 = dry, 1 = wet).
        """
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

    def get_color(self, alt: float, moisture: float) -> tuple[float, float, float]:
        """Return a smoothly blended biome RGB color for the given elevation and moisture.

        Ocean uses a depth-based gradient; land is looked up from the blurred biome table
        with bilinear interpolation.
        """
        if alt < self.water_level:
            depth = np.clip(
                (self.water_level - alt) / max(self.water_level, 1e-6), 0, 1
            )
            t = depth ** 0.7
            deep = np.array(self.BIOME_COLORS['DEEP_OCEAN'])
            shallow = np.array(self.BIOME_COLORS['SHALLOW_OCEAN'])
            return tuple(shallow * (1 - t) + deep * t)

        # Normalize land elevation into the LUT range
        e = np.clip((alt - self.water_level) / 0.25, 0, 1)
        m = np.clip(moisture, 0, 1)

        res = self.BIOME_TABLE_RES
        ei = min(int(e * (res - 1)), res - 1)
        mi = min(int(m * (res - 1)), res - 1)

        # Bilinear interpolation for extra-smooth transitions
        ef = e * (res - 1) - ei
        mf = m * (res - 1) - mi
        ei2 = min(ei + 1, res - 1)
        mi2 = min(mi + 1, res - 1)

        c00 = self._biome_table[ei, mi]
        c10 = self._biome_table[ei2, mi]
        c01 = self._biome_table[ei, mi2]
        c11 = self._biome_table[ei2, mi2]

        color = (c00 * (1 - ef) * (1 - mf) +
                 c10 * ef * (1 - mf) +
                 c01 * (1 - ef) * mf +
                 c11 * ef * mf)

        return tuple(color)

    def _get_colors_grid(self, heights: np.ndarray, moistures: np.ndarray) -> np.ndarray:
        """Vectorized biome coloring for full grids (used by ``export_colormap``)."""
        shape = heights.shape
        colors = np.zeros((*shape, 3))

        # Ocean - depth gradient from shallow to deep
        ocean = heights < self.water_level
        depth = np.clip(
            (self.water_level - heights) / max(self.water_level, 1e-6), 0, 1
        )
        t = depth ** 0.7
        deep = np.array(self.BIOME_COLORS['DEEP_OCEAN'])
        shallow = np.array(self.BIOME_COLORS['SHALLOW_OCEAN'])
        for c in range(3):
            colors[:, :, c] = np.where(
                ocean,
                shallow[c] * (1 - t) + deep[c] * t,
                0,
            )

        # Land - biome table lookup
        e = np.clip((heights - self.water_level) / 0.25, 0, 1)
        m = np.clip(moistures, 0, 1)

        res = self.BIOME_TABLE_RES
        ei = np.clip((e * (res - 1)).astype(int), 0, res - 1)
        mi = np.clip((m * (res - 1)).astype(int), 0, res - 1)

        land = ~ocean
        for c in range(3):
            colors[:, :, c] = np.where(
                land,
                self._biome_table[ei, mi, c],
                colors[:, :, c],
            )

        return colors

    @staticmethod
    def _resample_lanczos():
        """Return the Lanczos resampling constant (Pillow version compat)."""
        return Image.Resampling.LANCZOS

    def _interpolate_grid(self, values: np.ndarray, resolution: int,
                          fill_value: float) -> np.ndarray:
        """Cubic-interpolate per-region values onto a regular pixel grid."""
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        grid = griddata(
            self.points,
            values,
            grid_points,
            method="cubic",
            fill_value=fill_value,
        ).reshape(resolution, resolution)
        return np.flipud(grid)  # flip to match UV orientation

    def _apply_color_grade(self, colors: np.ndarray) -> np.ndarray:
        """Apply saturation/contrast/gamma grading for Blender-friendly albedo."""
        graded = np.clip(colors, 0, 1)

        gray = np.mean(graded, axis=2, keepdims=True)
        chroma = np.max(graded, axis=2, keepdims=True) - np.min(graded, axis=2, keepdims=True)
        vibrance_scale = 1.0 + self.COLOR_GRADE["vibrance"] * (1.0 - chroma)
        graded = gray + (graded - gray) * self.COLOR_GRADE["saturation"] * vibrance_scale

        graded = (graded - 0.5) * self.COLOR_GRADE["contrast"] + 0.5
        graded = np.clip(graded, 0, 1)
        graded = np.power(graded, self.COLOR_GRADE["gamma"])
        return np.clip(graded, 0, 1)

    def _rasterize_rivers(
        self,
        resolution: int,
        supersample: int = 2,
        width_scale: float = 1.0,
    ) -> np.ndarray:
        """Rasterize river segments into a float mask at the given resolution.

        Draws at *supersample*x resolution then downscales with Lanczos
        and a slight Gaussian blur for smooth anti-aliased edges.
        """
        canvas_size = int(resolution * supersample)
        mask_img = Image.new('L', (canvas_size, canvas_size), 0)
        draw = ImageDraw.Draw(mask_img)

        if not self.rivers:
            return np.zeros((resolution, resolution), dtype=float)

        # Scale width to average Voronoi cell size in export pixels.
        # This prevents "marker-thick" rivers on dense maps (20k/50k points).
        cell_px = resolution / np.sqrt(max(float(self.numRegions), 1.0))
        min_width_px = max(1, int(round(0.10 * cell_px * width_scale)))
        max_width_px = max(min_width_px + 1, int(round(0.32 * cell_px * width_scale)))

        for start, end, flow in self.rivers:
            x1 = int(start[0] / self.grid_size * (canvas_size - 1))
            y1 = int((1.0 - start[1] / self.grid_size) * (canvas_size - 1))
            x2 = int(end[0] / self.grid_size * (canvas_size - 1))
            y2 = int((1.0 - end[1] / self.grid_size) * (canvas_size - 1))

            t = (flow / self.max_flow) ** 0.45
            width_px = int(round(min_width_px + (max_width_px - min_width_px) * (t ** 0.75)))
            width = max(1, int(round(width_px * supersample)))

            draw.line([(x1, y1), (x2, y2)], fill=255, width=width)

        if supersample > 1:
            mask_img = mask_img.resize(
                (resolution, resolution),
                resample=self._resample_lanczos(),
            )

        river_mask = np.array(mask_img).astype(float) / 255.0
        river_mask = gaussian_filter(river_mask, sigma=0.45)  # light AA blur
        return np.clip(river_mask, 0, 1)

    def export_heightmap(self, filepath: str, resolution: int = 1024) -> None:
        """Write a 16-bit grayscale PNG heightmap."""
        heights = self._interpolate_grid(self.altitudes, resolution, fill_value=0)

        h_min, h_max = heights.min(), heights.max()
        if h_max - h_min > 0:
            heights = (heights - h_min) / (h_max - h_min)
        else:
            heights = np.zeros_like(heights)

        heights_16 = (heights * 65535).astype(np.uint16)
        img = Image.fromarray(heights_16, mode="I;16")
        img.save(filepath)

    def export_colormap(
        self,
        filepath: str,
        resolution: int = 1024,
        river_opacity: float = 0.75,
        river_width_scale: float = 1.0,
    ) -> None:
        """Write an RGB biome texture with rivers painted on top."""
        heights = self._interpolate_grid(self.altitudes, resolution, fill_value=0)
        moistures = self._interpolate_grid(self.moisture, resolution, fill_value=0.5)
        colors = self._get_colors_grid(heights, moistures)

        # Blend river color on top
        river_mask = self._rasterize_rivers(
            resolution,
            supersample=2,
            width_scale=river_width_scale,
        )
        river_blend = np.clip(river_mask * river_opacity, 0, 1)
        river_color = np.array(self.BIOME_COLORS['RIVER'])
        for c in range(3):
            colors[:, :, c] = (
                colors[:, :, c] * (1 - river_blend) +
                river_color[c] * river_blend
            )

        colors = self._apply_color_grade(colors)

        colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(colors_uint8, mode="RGB")
        img.save(filepath)

    def export_rivermap(
        self,
        filepath: str,
        resolution: int = 1024,
        river_width_scale: float = 1.0,
    ) -> None:
        """Write a grayscale river mask (white = river, black = land)."""
        river_mask = self._rasterize_rivers(
            resolution,
            supersample=2,
            width_scale=river_width_scale,
        )
        mask_uint8 = (river_mask * 255).astype(np.uint8)
        img = Image.fromarray(mask_uint8, mode="L")
        img.save(filepath)


def lloyd(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Run Lloyd relaxation to distribute points more evenly."""
    field = Field(points)
    for _ in range(iterations):
        field.relax()
    return field.get_points()