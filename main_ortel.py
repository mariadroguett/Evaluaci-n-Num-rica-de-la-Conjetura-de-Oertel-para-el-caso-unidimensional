# main_ortel.py
import argparse, os, csv
import numpy as np
from convex_hull import generate_convex_hull, random_vertices_by_fiber
from ortel import ortel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--n_per_z', type=int, default=15)
    p.add_argument('--d', type=int, default=2)
    p.add_argument('--n_point', type=int, default=5)
    p.add_argument('--z_vals', nargs='+', type=float, default=[0, 1, 2])
    p.add_argument('--N', type=int, default=300_000)
    p.add_argument('--N_cp', type=int, default=100)
    p.add_argument('--N_hip', type=int, default=1_000)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--target_mb', type=float, default=64.0)
    p.add_argument('--f_threshold', type=float, default=0.18)
    p.add_argument('--save_fibers_dir', type=str, default=None)
    p.add_argument('--save_hull_dir', type=str, default=None)
    p.add_argument('--out', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) Fibras (por z en z_vals, n_per_z puntos en [0,1]^d)
    verts = random_vertices_by_fiber(
        z_vals=args.z_vals, d=args.d, n_per_z=args.n_per_z,
        seed=args.seed, save_fibers_dir=args.save_fibers_dir
    )

    # 2) Hull en R^{1+d} y Ax<=b
    A, b = generate_convex_hull(verts, n_point=args.n_point)

    # 3) Experimento ORTEL (pasa batch/target_mb hasta rejection_sampling)
    bestF, bestCP = ortel(
        A, b, args.d,
        z_vals=args.z_vals,
        N_cp=args.N_cp,
        N_hip=args.N_hip,
        N=args.N,
        tol=1e-9,
        seed=args.seed,
        batch=args.batch,
        target_mb=args.target_mb,
    )

    print(f"Seed {args.seed}: BestCP = {bestCP}, F ~= {bestF}")

    # 4) Guardar hull si F < umbral
    if args.save_hull_dir and (bestF < args.f_threshold):
        os.makedirs(args.save_hull_dir, exist_ok=True)
        path = os.path.join(
            args.save_hull_dir, f"hull_seed_{args.seed}_npoint_{args.n_point}.npz"
        )
        np.savez(path, A=A, b=b, seed=args.seed, n_point=args.n_point, F=bestF)

    # 5) Guardar CSV (una fila)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["seed", "n_point", "F", "bestcp"])
            w.writerow([args.seed, args.n_point, bestF, bestCP.tolist()])

if __name__ == "__main__":
    main()
