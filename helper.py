import numpy as np
import opensimplex as osm
from lloyd import Field

def lloyd(points, iterations=1):
    """Perform Lloyd's relaxation on a set of points.
    
    Args:
        points (np.ndarray): An array of shape (N, 2) representing the initial points.
        iterations (int): The number of relaxation iterations to perform.   
        
    Returns:
        np.ndarray: The relaxed points after the specified number of iterations.    
    """
    field = Field(points)
    for i in range(iterations):
        field.relax()
        
    return field.get_points()

def circumCenter(a, b, c):
    """Calculate the circumcenter of a triangle given by points a, b, and c.
    
    Args:
        a, b, c (np.ndarray): Arrays of shape (2,) representing the triangle vertices.
        
    Returns:
        np.ndarray: An array of shape (2,) representing the circumcenter of the triangle.
    """
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
    uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
    return np.array([ux, uy])

def next_halfedge(e: int) -> int:
    """Index of the next halfedge within the same triangle."""
    return e - e % 3 + (e + 1) % 3

def triangle_of_edge(e: int) -> int:
    """Triangle index for halfedge e."""
    return e // 3

def edges_around_point(halfedges, triangles, start_e):
    """
    Collect halfedges circulating around the site triangles[start_e].
    """
    # Walk around the site (JS-equivalent) until we return to start or hit hull
    result = []
    incoming = start_e
    while True:
        result.append(incoming)
        outgoing = next_halfedge(incoming)
        incoming = halfedges[outgoing]
        if incoming == -1 or incoming == start_e:
            break
    return result

def assignElevation(water_level, points, width=1.0):
    """Assign elevation values based on distance from center plus layered noise.
    Accepts any array of 2D coordinates (sites or centers)."""
    n = len(points)
    elevation = np.zeros(n, dtype=float)
    for i in range(n):
        nx = points[i][0] / width - 0.5
        ny = points[i][1] / width - 0.5
        # Layered OpenSimplex noise with different frequencies & amplitudes
        base = (
            0.5
            + osm.noise2(nx / water_level, ny / water_level) / 2.0
            + osm.noise2(nx * 2 / water_level, ny * 2 / water_level) / 5.0
        )
        # Island falloff using Chebyshev distance (square-ish) from center
        d = 2 * max(abs(nx), abs(ny))
        elevation[i] = (1 + base - d) / 2.0
    return elevation