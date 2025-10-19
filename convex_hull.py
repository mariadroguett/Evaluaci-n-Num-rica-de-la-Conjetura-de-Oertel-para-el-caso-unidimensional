import os
import csv
import numpy as np
from scipy.spatial import ConvexHull


def random_vertices_by_fiber(z_vals, d=2, n_per_z=30, seed=None, save_fibers_dir=None, return_by_fiber=False):
    """
    Genera puntos aleatorios (vértices candidatos) en R^{1+d}:
      - para cada z en z_vals, genera n_per_z puntos p ~ U([0,1]^d)
      - devuelve un array (len(z_vals)*n_per_z, 1+d) con primera col = z
    Opcionalmente, guarda las fibras en CSV y/o retorna un dict por fibra.
    """
    rng = np.random.default_rng(seed)
    all_pts = []
    by_fiber = {}
    for z in z_vals:
        P = rng.random((n_per_z, d))                        # (n_per_z, d)
        Z = np.full((n_per_z, 1), float(int(z)), float)     # (n_per_z, 1)
        V = np.hstack([Z, P])                               # (n_per_z, 1+d)
        all_pts.append(V)
        by_fiber[int(z)] = P.copy()

    verts = np.vstack(all_pts)

    if save_fibers_dir:
        try:
            os.makedirs(save_fibers_dir, exist_ok=True)
            # archivo completo con z,p1..pd
            all_path = os.path.join(save_fibers_dir, "all_points.csv")
            header_all = ["z"] + [f"p{i+1}" for i in range(d)]
            with open(all_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header_all)
                for row in verts:
                    w.writerow([int(row[0])] + [float(x) for x in row[1:1+d]])

            # archivos por fibra: fiber_z_<z>.csv con p1..pd
            for z, P in by_fiber.items():
                z_path = os.path.join(save_fibers_dir, f"fiber_z_{z}.csv")
                header_z = [f"p{i+1}" for i in range(d)]
                with open(z_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(header_z)
                    for p in P:
                        w.writerow([float(x) for x in p])
        except Exception:
            # No interrumpir si falla el guardado
            pass

    if return_by_fiber:
        return verts, by_fiber
    return verts


def generate_convex_hull(verts, tol=1e-10, n_point=5, qhull_opts="QJ"):
    """
    Calcula la envolvente convexa de un conjunto de vértices y devuelve la
    descripción Ax ≤ b, donde A y b permanecen inmutables en el resto del código.

    Parámetros
    ----------
    verts : array (m, d_tot)
        Coordenadas de los puntos (cada fila es z + variables continuas).
    tol : float
        Tolerancia para decidir de qué lado queda un punto interior.
    dedupe_decimals : int
        Decimales usados para redondear y eliminar hiperplanos duplicados.
    qhull_opts : str
        Parámetros de Qhull (por ejemplo "QJ" para pequeños perturbados).

    Retorna
    -------
    A : array (k, d_tot)
    b : array (k,)
        Tal que la envolvente convexa es { x : A x ≤ b }.
    """
    V = np.asarray(verts, dtype=float)
    if V.ndim != 2 or V.shape[0] < V.shape[1] + 1:
        raise ValueError("Debe haber al menos d_tot+1 vértices no coplanares.")

    hull = ConvexHull(V, qhull_options=qhull_opts)
    interior = V.mean(axis=0)  # punto interior aproximado

    A_list, b_list = [], []
    for eq in hull.equations:
        n = eq[:-1].astype(float)
        c = float(eq[-1])
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            continue
        n /= norm
        c /= norm

        # Orientar las desigualdades hacia el interior (n·x + c ≤ 0).
        if (n @ interior + c) <= tol:
            A_list.append(n.copy())
            b_list.append(-c)
        else:
            A_list.append((-n).copy())
            b_list.append(c)

    # Eliminar planos duplicados aproximando sus coeficientes.
    seen = set()
    A_clean, b_clean = [], []
    for a, bi in zip(A_list, b_list):
        key = (tuple(np.round(a, n_point)), float(np.round(bi, n_point)))
        if key not in seen:
            seen.add(key)
            A_clean.append(a)
            b_clean.append(bi)

    A = np.vstack(A_clean) if A_clean else np.zeros((0, V.shape[1]), dtype=float)
    b = np.array(b_clean, dtype=float)
    return A, b


#Nota: El hecho de que salgan cosas como -0. o normales con componentes como 0.8944... 
# es porque SciPy normaliza automáticamente los vectores normales a norma 1.

#El orden de las filas de A y b no necesariamente coincide con el orden de los vértices: 
# ConvexHull las ordena según su propia lógica interna para recorrer las caras.
    
