#!/usr/bin/env python3
# main_ortel.py (seedless, NPZ-only, routing por F)
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np

from convex_hull import random_vertices_by_fiber, generate_convex_hull
from ortel import ortel  # tu versión sin seed


def build_parser():
    p = argparse.ArgumentParser(
        description="Experimento Oertel (seedless) → guarda NPZ con A, b, F, bestcp y vértices, separando entre hulls y hulls_obs según F."
    )
    p.add_argument("--d", type=int, required=True, help="dimensión continua")
    p.add_argument("--z_vals", nargs="+", type=int, required=True, help="fibras z (enteras)")
    p.add_argument("--n_per_z", type=int, default=5, help="puntos por fibra (por defecto)")
    p.add_argument("--n_point", type=int, default=None, help="override de puntos por fibra (si se da, manda sobre n_per_z)")
    p.add_argument("--N", type=int, required=True, help="muestras Monte Carlo por z")
    p.add_argument("--N_cp", type=int, required=True, help="candidatos de centerpoint")
    p.add_argument("--N_hip", type=int, required=True, help="hiperplanos para evaluar el peor corte")
    p.add_argument("--f_threshold", type=float, default=0.18, help="umbral F para enrutar hulls vs hulls_obs")
    p.add_argument("--target_mb", type=float, default=None, help="MiB objetivo para batches internos")
    p.add_argument("--results_root", type=Path, default=Path("results"), help="carpeta raíz para guardar resultados")

    # flags legacy (compatibilidad)
    p.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--save_hull_dir", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--save_hull_obs_dir", type=Path, default=None, help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()

    d = int(args.d)
    z_vals = [int(z) for z in args.z_vals]
    n_per_z = int(args.n_point) if args.n_point is not None else int(args.n_per_z)
    N = int(args.N)
    N_cp = int(args.N_cp)
    N_hip = int(args.N_hip)
    f_threshold = float(args.f_threshold)
    target_mb = args.target_mb

    # fecha/timestamp
    day_str = datetime.now().strftime("%Y-%m-%d")

    # 1) puntos por fibra
    verts = random_vertices_by_fiber(z_vals, d, n_per_z)

    # 2) envolvente convexa
    A, b = generate_convex_hull(verts)

    # 3) búsqueda de centerpoint
    bestCP, bestF = ortel(
        A, b, d,
        z_vals=z_vals,
        N_cp=N_cp,
        N_hip=N_hip,
        N=N,
        tol=1e-9,
        batch=None,
        target_mb=target_mb,
    )

    # 4) ruta de guardado según F
    subdir = "hulls" if bestF >= f_threshold else "hulls_obs"
    day_dir = args.results_root / subdir / day_str
    day_dir.mkdir(parents=True, exist_ok=True)

    # timestamp con fecha+hora+minuto+segundo
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # base del nombre del archivo
    base = f"npoint_{n_per_z}_{ts}"

    # rutas completas dentro de la carpeta del día
    result_path = f"{day_dir}/result_{base}.npz"
    verts_path  = f"{day_dir}/verts_{base}.npz"
    # 5) guardar NPZs — ¡sin np.string_()!
    np.savez_compressed(
        result_path,
        A=A,
        b=b,
        F=np.float64(bestF),
        bestcp=np.asarray(bestCP, dtype=float),
        d=np.int64(d),
        z_vals=np.array(z_vals, dtype=np.int64),
        N=np.int64(N),
        N_cp=np.int64(N_cp),
        N_hip=np.int64(N_hip),
        n_per_z=np.int64(n_per_z),
        f_threshold=np.float64(f_threshold),
        target_mb=(np.float64(target_mb) if target_mb is not None else np.float64(np.nan)),
        timestamp=np.int64(ts),
        saved_dir=str(day_dir),
        file_tag=base,
    )

    np.savez_compressed(
        verts_path,
        verts=verts,
        d=np.int64(d),
        z_vals=np.array(z_vals, dtype=np.int64),
        n_per_z=np.int64(n_per_z),
        timestamp=np.int64(ts),
        saved_dir=str(day_dir),
        file_tag=base,
    )

    print(f"[OK] F={bestF:.5f} (threshold={f_threshold}) -> {subdir}\n"
          f"  - {result_path}\n"
          f"  - {verts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
