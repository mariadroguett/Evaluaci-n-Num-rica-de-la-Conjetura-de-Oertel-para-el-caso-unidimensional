import numpy as np
from scipy.spatial import ConvexHull

def random_vertices_by_fiber(d, z_vals, n_per_z, seed=None):
    """
    Genera vértices aleatorios por fibra z en [0,1]^d.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    vertices = []
    for z in z_vals:
        pts = rng.random((n_per_z, d))
        verts_z = np.column_stack([np.full(n_per_z, z), pts])
        vertices.append(verts_z)

    return np.vstack(vertices)

def generate_convex_hull(verts):
    """Devuelve A,b de la envolvente convexa."""
    hull = ConvexHull(verts)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    return A, b
