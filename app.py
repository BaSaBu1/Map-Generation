"""
Procedural Map Generator - Streamlit Web Application.

Interactive web-based terrain generation using Voronoi diagrams and Perlin noise.
Deployed at: https://basabu1-map-generation-app-fut9qy.streamlit.app/

Usage:
    streamlit run app.py

Author: Batsambuu Batbold
Course: MATH 437 - Computational Geometry
Date: December 2025
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from map import Map


st.set_page_config(
    page_title="Procedural Map Generator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; }
        h1 { color: #2E4057; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def generate_map_figure(
    seed: int,
    num_points: int = 2000,
    noise_scale: float = 4.0,
    water_level: float = 0.35,
    clusters: int = 5,
) -> plt.Figure:
    """
    Generate terrain and return a matplotlib figure.
    
    Args:
        seed: Random seed for reproducibility.
        num_points: Number of Voronoi sites.
        noise_scale: Perlin noise frequency (1-10).
        water_level: Ocean/land threshold (0-0.8).
        clusters: Number of island centers.
        
    Returns:
        Matplotlib figure containing the rendered terrain.
    """
    np.random.seed(seed)
    points = np.random.rand(num_points, 2)
    terrain = Map(
        points,
        size=1,
        water_level=water_level,
        noise_scale=noise_scale,
        cluster=clusters,
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    terrain.plotLand(ax)
    plt.tight_layout(pad=0)

    return fig


def main() -> None:
    """Main application entry point."""
    st.title("🗺️ Procedural Map Generator")
    st.markdown(
        "*Generate unique worlds using Voronoi diagrams, Lloyd's relaxation, and Perlin noise*"
    )
    
    # Info expander
    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        This application generates procedural terrain maps using computational geometry:
        
        - **Voronoi Diagrams**: Partition space into regions
        - **Lloyd's Relaxation**: Creates uniform point distribution
        - **Perlin Noise**: Generates natural-looking elevation
        - **Biome System**: 10 distinct biomes based on elevation and moisture
        """)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Map Controls")

        seed = st.number_input(
            "🎲 Random Seed",
            value=42,
            min_value=0,
            max_value=9999,
            help="Change this for a completely different map",
        )

        st.divider()

        noise_scale = st.slider(
            "🔍 Noise Scale",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help="Higher = more detailed, but chaotic terrain features",
        )

        water_level = st.slider(
            "🌊 Water Level",
            min_value=0.0,
            max_value=0.8,
            value=0.35,
            step=0.05,
            help="Higher = more ocean, less land",
        )

        clusters = st.slider(
            "🏝️ Island Clusters",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of landmass centers",
        )

        num_points = st.select_slider(
            "📍 Resolution",
            options=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
            value=2000,
            help="More points = finer detail (slower)",
        )

        st.divider()
        st.button("🔄 Generate New Map", type="primary", use_container_width=True)
        st.divider()

        st.caption("MATH 437 | Computational Geometry")
        st.caption("Batsambuu Batbold | December 2025")

    # Map display
    try:
        with st.spinner("🌍 Generating terrain..."):
            fig = generate_map_figure(
                seed=seed,
                num_points=num_points,
                noise_scale=noise_scale,
                water_level=water_level,
                clusters=clusters,
            )
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig, use_container_width=True)
            
            # Performance info
            st.caption(f"✓ Generated with {num_points:,} points | Seed: {seed}")
            
            plt.close(fig)
    except Exception as e:
        st.error(f"❌ Error generating map: {str(e)}")
        st.info("Try adjusting the parameters or using a different seed.")


if __name__ == "__main__":
    main()
