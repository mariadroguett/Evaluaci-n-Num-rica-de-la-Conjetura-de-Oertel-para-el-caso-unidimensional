#!/usr/bin/env python3
import argparse
import numpy as np
import os
import csv

from convex_hull import generate_convex_hull, random_vertices_by_fiber
from ortel import ortel


def parse_args():
    p = argparse.ArgumentParser("Ejecuta un experimento Oertel para un (seed, n_point).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--z_vals", nargs="+", type=float, default=[0, 1, 2])
    p.add_argument("--n_per_z", type=int, default=5, help="puntos por fibra (por cada z)")
    p.add_argument("--n_point", type=int, default=5, help="solo para redondeo/deduplicación de planos del hull")
    p.add_argument("--N", type=int, default=300_000)
    p.add_argument("--N_cp", type=int, default=100)
    p.add_argument("--N_hip", type=int, default=1000)
    p.add_argument("--f_threshold", type=float, default=0.18)
    p.add_argument("--out", type=str, default=None, help="CSV de salida donde se appendea 1 fila por corrida")
    p.add_argument("--save_hull_dir", type=str, default=None, help="carpeta para guardar hulls cuando F ≥ threshold")
    p.add_argument("--save_hull_obs_dir", type=str, default=None, help="carpeta para guardar hulls cuando F < threshold")

    # 👇 agregado: solo para que no falle si lo pasas desde el runner
    p.add_argument("--target_mb", type=float, default=64.0,
                   help="Memoria objetivo (MB) para batching en rejection_sampling (si tu ortel/vol_reject lo usa).")
    # (Si tu ortel.py no lo usa, no pasa nada. Evitamos el error de 'unrecognized arguments'.)

    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) Vértices aleatorios por fibra (no pasamos kwargs raros para evitar incompatibilidades)
    #    Firma esperada en tu repo: random_vertices_by_fiber(z_vals, d=2, n_per_z=30, seed=None, save_fibers_dir=None, return_by_fiber=False)
    try:
        verts = random_vertices_by_fiber(args.d, args.z_vals, n_per_z=args.n_per_z, seed=args.seed)
    except TypeError:
        # Si tu versión en el cluster NO acepta 'seed', reintenta sin él
        verts  = random_vertices_by_fiber(args.d ,args.z_vals, n_per_z=args.n_per_z)

    # 2) Hull en R^{1+d}
    A, b = generate_convex_hull(verts)

    # 3) Correr Oertel (usa tu ortel.py tal cual)
    #    Nota: no pasamos target_mb aquí por si tu ortel.py no lo acepta.
    bestCP, bestF = ortel(
        A, b, args.d,
        z_vals=args.z_vals,
        N_cp=args.N_cp,
        N_hip=args.N_hip,
        N=args.N,
        tol=1e-9,
        seed=args.seed,
        batch=None,
    )

    print(f"Seed {args.seed}: BestCP = {bestCP}, F ~= {bestF}")

    # 4) Guardar hull según umbral
    #    Regla: F ≥ threshold  → save_hull_dir
    #           F < threshold  → save_hull_obs_dir
    if bestF >= args.f_threshold and args.save_hull_dir:
        os.makedirs(args.save_hull_dir, exist_ok=True)
        npz_path = os.path.join(
            args.save_hull_dir, f"hull_seed_{args.seed}_npoint_{args.n_point}.npz"
        )
        np.savez(npz_path, A=A, b=b, seed=args.seed, n_point=args.n_point, F=bestF)
    elif bestF < args.f_threshold and args.save_hull_obs_dir:
        os.makedirs(args.save_hull_obs_dir, exist_ok=True)
        npz_path = os.path.join(
            args.save_hull_obs_dir, f"hull_seed_{args.seed}_npoint_{args.n_point}.npz"
        )
        np.savez(npz_path, A=A, b=b, seed=args.seed, n_point=args.n_point, F=bestF)

    # 5) Escribir 1 fila en CSV (append, con header si no existe)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["seed", "n_point", "F", "bestcp"])
            w.writerow([args.seed, args.n_point, bestF, bestCP.tolist()])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
