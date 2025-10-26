#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import csv
import sys
import numpy as np

from convex_hull import generate_convex_hull, random_vertices_by_fiber
from ortel import ortel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--n_per_z', type=int, default=15)
    p.add_argument('--d', type=int, default=2)
    # n_point en generate_convex_hull controla el redondeo para deduplicar planos
    p.add_argument('--n_point', type=int, default=5)
    p.add_argument('--z_vals', nargs='+', type=float, default=[0, 1])

    p.add_argument('--N', type=int, default=300_000)
    p.add_argument('--N_cp', type=int, default=100)
    p.add_argument('--N_hip', type=int, default=1_000)

    # batch es opcional; si tu ortel/vol_reject lo usa, cámbialo aquí
    p.add_argument('--batch', type=int, default=None)

    p.add_argument('--f_threshold', type=float, default=0.18)
    p.add_argument('--save_hull_dir', type=str, default=None)
    p.add_argument('--out', type=str, default=None)  # CSV de salida (1 fila por corrida)
    return p.parse_args()


def _unique_path(base_path: str) -> str:
    """
    Si base_path existe, agrega sufijos _1, _2, ... hasta encontrar uno libre.
    """
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    k = 1
    while True:
        cand = f"{root}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    # Forzamos UTF-8 en Windows/Linux para que no se caiga por símbolos raros
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) Generar puntos por fibra (z_vals) y construir hull
    verts = random_vertices_by_fiber(
        z_vals=args.z_vals,
        d=args.d,
        n_per_z=args.n_per_z,
        seed=args.seed,
        save_fibers_dir=None
    )

    A, b = generate_convex_hull(verts, n_point=args.n_point)

    # 2) ORTEL
    ortel_kwargs = dict(
        A=A,
        b=b,
        d=args.d,
        z_vals=args.z_vals,
        N_cp=args.N_cp,
        N_hip=args.N_hip,
        N=args.N,
        tol=1e-9,
        seed=args.seed
    )
    if args.batch is not None:
        ortel_kwargs["batch"] = args.batch

    bestCP, bestF = ortel(**ortel_kwargs)

    # Prints simples (sin caracteres unicode raros)
    print(f"Seed {args.seed}: BestCP = {np.array(bestCP)}, F ~= {float(bestF)}")

    # 3) Guardar hull si F < umbral, con nombre único
    if args.save_hull_dir and (bestF < args.f_threshold):
        os.makedirs(args.save_hull_dir, exist_ok=True)
        base = os.path.join(
            args.save_hull_dir,
            f"hull_seed_{args.seed}_npoint_{args.n_point}_F{bestF:.6f}.npz"
        )
        path = _unique_path(base)
        np.savez(
            path,
            A=A,
            b=b,
            seed=args.seed,
            n_point=args.n_point,
            F=float(bestF)
        )

    # 4) Escribir 1 fila en el CSV de salida (si se indicó --out)
    if args.out:
        _ensure_parent_dir(args.out)
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                # columnas explícitas → fácil de analizar luego
                headers = ["seed", "n_point", "F"] + [f"bestcp_{i}" for i in range(len(bestCP))]
                w.writerow(headers)
            row = [args.seed, args.n_point, float(bestF)] + list(np.asarray(bestCP, dtype=float))
            w.writerow(row)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("Interrumpido por usuario", file=sys.stderr)
        sys.exit(130)
