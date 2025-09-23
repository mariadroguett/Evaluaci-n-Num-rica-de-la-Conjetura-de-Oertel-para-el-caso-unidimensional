from convex_hull import generate_convex_hull,random_vertices_by_fiber
from ortel import ortel
import numpy as np

def main():
    # ----------------------------
    # 1) Politopo aleatorio
    # ----------------------------
    d = 2
    z_vals = [0, 1]
    d = 2
    z_vals = [0, 1]
    # Guarda las fibras para poder reutilizarlas después
    verts = random_vertices_by_fiber(z_vals, d=d, n_per_z=15, seed=123, save_fibers_dir="results/fibras")
    A, b = generate_convex_hull(verts)

    bestCP, bestF = ortel(
        A, b, d,
        z_vals=z_vals,
        N_cp=50,
        N_hip=1000,
        N=3*10**5,
        tol=1e-9,
        seed=None,
        batch=5000,    # controla memoria
        save_dir="results/cp_aceptados"
    )
    print("BestCP:", bestCP)
    print("F(S) ≈", bestF)


if __name__ == "__main__":
    main()
