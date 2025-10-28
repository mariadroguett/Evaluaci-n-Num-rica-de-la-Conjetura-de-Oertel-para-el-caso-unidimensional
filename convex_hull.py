import numpy as np
from scipy.spatial import ConvexHull

def random_vertices_by_fiber(d, z_vals, n_per_z):
    """Genera vértices aleatorios por fibra z."""
    all_points = []
    for z in z_vals:
        pts = np.random.rand(n_per_z, d)
        pts_z = np.column_stack((np.full(n_per_z, z), pts))
        all_points.append(pts_z)
    return np.vstack(all_points)

def generate_convex_hull(points):
    """Devuelve A,b de la envolvente convexa."""
    hull = ConvexHull(points)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    return A, b
