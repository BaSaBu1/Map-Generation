"""Streamlit web UI for interactive terrain generation.

Run with ``streamlit run app.py`` to explore parameters, compare seeds,
and preview maps before exporting textures.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore[reportMissingImports]
    HAS_DRAWABLE_CANVAS = True
except Exception:
    st_canvas = None
    HAS_DRAWABLE_CANVAS = False

from map import Map


DEFAULTS = {
    "seed": 42,
    "noise_scale": 4.0,
    "water_level": 0.35,
    "land_centers": 5,
    "show_rivers": True,
    "num_points": 2000,
    "land_layout": "Random",
    "layout_preset": "4 Corners",
    "custom_anchors": [],
    "custom_canvas_nonce": 0,
}

LAYOUT_PRESETS = {
    "4 Corners": [(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)],
    "Center Island": [(0.5, 0.5)],
    "Two Continents (E-W)": [(0.25, 0.5), (0.75, 0.5)],
    "Two Continents (N-S)": [(0.5, 0.25), (0.5, 0.75)],
    "Archipelago Ring": [(0.5, 0.15), (0.85, 0.5), (0.5, 0.85), (0.15, 0.5)],
}


st.set_page_config(
    page_title="Procedural Map Generator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _ensure_state_defaults() -> None:
    """Populate session state with defaults if not already set."""
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _randomize_seed() -> None:
    """Assign a random seed for quick map exploration."""
    st.session_state.seed = int(np.random.default_rng().integers(0, 10000))


def _reset_controls() -> None:
    """Restore all sidebar controls to their defaults."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def _extract_canvas_anchors(canvas_result, max_points: int = 5) -> list[tuple[float, float]]:
    """Read point objects from the drawing canvas and normalize to [0, 1]."""
    anchors: list[tuple[float, float]] = []
    if not canvas_result or not canvas_result.json_data:
        return anchors

    objects = canvas_result.json_data.get("objects", [])
    for obj in objects:
        if obj.get("type") != "circle":
            continue

        # Fabric circles store the top-left corner and radius in canvas pixels.
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        radius = float(obj.get("radius", 0.0))
        cx = left + radius
        cy = top + radius

        x_norm = float(np.clip(cx / 300.0, 0.0, 1.0))
        y_norm = float(np.clip(1.0 - (cy / 300.0), 0.0, 1.0))
        anchors.append((x_norm, y_norm))

    return anchors[:max_points]


@st.cache_data(show_spinner=False)
def generate_map_figure(
    seed: int,
    num_points: int = 2000,
    noise_scale: float = 4.0,
    water_level: float = 0.35,
    land_centers: int = 5,
    show_rivers: bool = True,
    custom_anchors: tuple[tuple[float, float], ...] | None = None,
) -> Figure:
    """Build and render a terrain figure. Cached by Streamlit on all args."""
    rng = np.random.default_rng(seed)
    points = rng.random((num_points, 2))
    terrain = Map(
        points,
        size=1,
        water_level=water_level,
        noise_scale=noise_scale,
        land_centers=land_centers,
        custom_anchors=list(custom_anchors) if custom_anchors else None,
        seed=seed,
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    terrain.plotLand(ax)
    if show_rivers:
        terrain.plotRivers(ax)
    plt.tight_layout(pad=0)

    return fig


def main() -> None:
    """Layout the Streamlit UI and render the current map preview."""
    _ensure_state_defaults()

    st.title("🗺️ Procedural Map Generator")
    st.markdown(
        "*Generate unique worlds using Voronoi diagrams, Lloyd's relaxation, and Perlin noise*"
    )
    
    # Info expander
    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        Procedural terrain generation using computational geometry:
        
        - **Voronoi Diagrams** - partition space into regions
        - **Lloyd's Relaxation** - create uniform point distributions
        - **Perlin Noise** - natural-looking elevation
        - **Biome System** - 10 biomes with Gaussian-blended transitions
        - **River Network** - hydrological flow along Voronoi edges
        """)

    with st.sidebar:
        st.header("⚙️ Map Controls")

        # Quick-action buttons for seed scanning
        controls_col1, controls_col2 = st.columns(2)
        controls_col1.button("🎲 Random", use_container_width=True, on_click=_randomize_seed)
        controls_col2.button("↩ Reset", use_container_width=True, on_click=_reset_controls)

        seed = st.number_input(
            "🎲 Random Seed",
            value=st.session_state.seed,
            min_value=0,
            max_value=9999,
            key="seed",
            help="Change this for a completely different map",
        )

        st.divider()

        noise_scale = st.slider(
            "🔍 Noise Scale",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.noise_scale,
            step=0.5,
            key="noise_scale",
            help="Higher = more detailed, but chaotic terrain features",
        )

        water_level = st.slider(
            "🌊 Water Level",
            min_value=0.0,
            max_value=0.8,
            value=st.session_state.water_level,
            step=0.05,
            key="water_level",
            help="Higher = more ocean, less land",
        )

        anchor_mode = st.radio(
            "📍 Land Layout",
            ["Random", "Preset", "Custom"],
            key="land_layout",
            horizontal=True,
            help="Random scatters anchors. Preset uses pre-defined layouts. Custom lets you click anchors.",
        )

        if anchor_mode == "Random":
            land_centers = st.slider(
                "🏔️ Land Centers",
                min_value=1,
                max_value=10,
                value=st.session_state.land_centers,
                key="land_centers",
                help="More anchor points usually create more distinct landmasses",
            )
            custom_anchors = None
        elif anchor_mode == "Preset":
            layout_preset = st.selectbox(
                "🗺️ Layout Preset",
                options=list(LAYOUT_PRESETS.keys()),
                key="layout_preset",
                help="Choose a pre-defined anchor arrangement",
            )
            custom_anchors = tuple((p[0], p[1]) for p in LAYOUT_PRESETS[layout_preset])
            land_centers = len(custom_anchors)
        else:
            st.caption("Click on the white window below to add up to 5 custom points as places for land anchors.")
            clear_points = st.button("🧹 Clear Custom Anchors", use_container_width=True)

            if clear_points:
                st.session_state.custom_anchors = []
                st.session_state.custom_canvas_nonce += 1

            if HAS_DRAWABLE_CANVAS and st_canvas is not None:
                canvas_result = st_canvas(
                    fill_color="rgba(30, 136, 229, 0.85)",
                    stroke_width=1,
                    stroke_color="#1565c0",
                    background_color="#f6f8fb",
                    update_streamlit=True,
                    height=300,
                    width=300,
                    drawing_mode="point",
                    point_display_radius=6,
                    key=f"anchor_canvas_{st.session_state.custom_canvas_nonce}",
                )

                extracted = _extract_canvas_anchors(canvas_result, max_points=5)
                if extracted:
                    st.session_state.custom_anchors = extracted
            else:
                st.warning(
                    "Interactive canvas unavailable. Install streamlit-drawable-canvas to enable click placement."
                )

            custom_anchor_list = list(st.session_state.custom_anchors)
            if len(custom_anchor_list) >= 5:
                st.caption("Maximum reached: 5 anchors")

            if custom_anchor_list:
                st.caption(
                    "Selected anchors: " + ", ".join(
                        [f"({x:.2f}, {y:.2f})" for x, y in custom_anchor_list]
                    )
                )
                custom_anchors = tuple(custom_anchor_list)
                land_centers = len(custom_anchors)
            else:
                st.info("Add at least 1 anchor point to use Custom mode.")
                # Fail-safe so generation still works if user has not clicked yet.
                custom_anchors = None
                land_centers = st.session_state.land_centers

        show_rivers = st.checkbox(
            "🏞️ Show Rivers",
            value=st.session_state.show_rivers,
            key="show_rivers",
            help="Overlay hydrological river network",
        )

        num_points = st.select_slider(
            "📍 Resolution",
            options=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
            value=st.session_state.num_points,
            key="num_points",
            help="More points = finer detail (slower)",
        )

        st.divider()
        st.caption("Batsambuu Batbold | Macalester College")

    try:
        with st.spinner("🌍 Generating terrain..."):
            fig = generate_map_figure(
                seed=seed,
                num_points=num_points,
                noise_scale=noise_scale,
                water_level=water_level,
                land_centers=land_centers,
                show_rivers=show_rivers,
                custom_anchors=custom_anchors,
            )
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig, use_container_width=True)
            
            st.caption(f"✓ Generated with {num_points:,} points | Seed: {seed}")
            
            plt.close(fig)
    except Exception as e:
        st.error(f"❌ Error generating map: {str(e)}")
        st.info("Try adjusting the parameters or using a different seed.")


if __name__ == "__main__":
    main()
