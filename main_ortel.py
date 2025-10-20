import argparse
import numpy as np
import os
import csv
import sys
from convex_hull import generate_convex_hull, random_vertices_by_fiber
from ortel import ortel

# --- Windows-friendlya ---
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n_per_z', type=int, default=15)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--n_point', type=int, default=5)
    parser.add_argument('--z_vals', nargs='+', type=float, default=[0, 1])
    parser.add_argument('--N', type=int, default=3*10**5)
    parser.add_argument('--N_cp', type=int, default=100)
    parser.add_argument('--N_hip', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=5000)
    parser.add_argument('--triangle', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--f_threshold', type=float, default=0.18)
    parser.add_argument('--save_fibers_dir', type=str, default=None)
    parser.add_argument('--save_hull_dir', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Genera vértices por fibra (usa firma de tu convex_hull.py)
    verts = random_vertices_by_fiber(
        z_vals=args.z_vals, d=args.d, n_per_z=args.n_per_z,
        seed=args.seed, save_fibers_dir=args.save_fibers_dir
    )

    d = args.d
    triangle = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1/2, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1/2, 1]
    ], dtype=float)

    # n_point aquí solo se usa como precisión de redondeo en generate_convex_hull
    A, b = generate_convex_hull(triangle, n_point=args.n_point)

    bestCP, bestF = ortel(
        A, b, d,
        z_vals=args.z_vals,
        N_cp=args.N_cp,
        N_hip=args.N_hip,
        N=args.N,
        tol=1e-9,
        seed=args.seed,
        batch=args.batch
    )

    # Evitar símbolo ≈ en Windows
    print(f"Seed {args.seed}: BestCP = {bestCP}, F ~= {bestF}")

    # Guardar hull si F < threshold
    if args.save_hull_dir and bestF < args.f_threshold:
        os.makedirs(args.save_hull_dir, exist_ok=True)
        hull_path = os.path.join(
            args.save_hull_dir,
            f"hull_seed_{args.seed}_npoint_{args.n_point}.npz"
        )
        np.savez(
            hull_path,
            A=A,
            b=b,
            seed=args.seed,
            n_point=args.n_point,
            F=bestF
        )

    # Guardar resultados individuales
    if args.out:
        write_header = not os.path.exists(args.out)
        # CSV en UTF-8 para no pelear con acentos
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["n_point", "F", "bestcp"])
            writer.writerow([args.n_point, bestF, bestCP.tolist()])

if __name__ == "__main__":
    main()
