import argparse
from convex_hull import generate_convex_hull, random_vertices_by_fiber
from ortel import ortel
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Ejecuta ORTEL con opciones de baúl y paralelización")
    parser.add_argument("--d", type=int, default=2, help="Dimensión d del cubo [0,1]^d")
    parser.add_argument("--z_vals", type=str, default="0,1", help="Lista de fibras z separadas por coma")
    parser.add_argument("--n_per_z", type=int, default=15, help="Vértices por fibra para generar el hull")
    parser.add_argument("--seed", type=lambda s: None if s=="None" else int(s), default=123, help="Semilla RNG para generar vértices")

    # ORTEL/ratio
    parser.add_argument("--N_cp", type=int, default=50, help="Número de candidatos de centerpoint")
    parser.add_argument("--N_hip", type=int, default=100, help="Número de direcciones para cortes en ratio_cp")
    parser.add_argument("--N", type=int, default=100000, help="Tamaño de ronda/base para muestreo (N o chunk_N)")
    parser.add_argument("--tol", type=float, default=1e-9, help="Tolerancia Ax <= b")
    parser.add_argument("--guided", action="store_true", help="Modo guided en ratio_cp (si aplica)")

    # Baúl
    parser.add_argument("--into_baul", type=int, default=5000, help="Aceptados objetivo por fibra para estimar volumen")
    parser.add_argument("--use_parallel", action="store_true", help="Usar baúl paralelo")
    parser.add_argument("--chunk_N", type=int, default=None, help="Tamaño de chunk en paralelo (por defecto = N)")
    parser.add_argument("--max_workers", type=int, default=None, help="Procesos paralelos")
    parser.add_argument("--max_draws", type=int, default=None, help="Límite total de sorteos por estimación")

    args = parser.parse_args()

    d = int(args.d)
    z_vals = [int(s.strip()) for s in args.z_vals.split(",") if s.strip()]

    # 1) Politopo aleatorio a partir de vértices por fibra
    verts = random_vertices_by_fiber(z_vals, d=d, n_per_z=int(args.n_per_z), seed=args.seed)
    A, b = generate_convex_hull(verts)

    # 2) Ejecutar ORTEL
    bestCP, bestF = ortel(
        A, b, d,
        z_vals=z_vals,
        N_cp=int(args.N_cp),
        N_hip=int(args.N_hip),
        N=int(args.N),
        tol=float(args.tol),
        seed=None,
        batch=None,
        guided=bool(args.guided),
        into_baul=int(args.into_baul),
        use_parallel=bool(args.use_parallel),
        chunk_N=(int(args.chunk_N) if args.chunk_N is not None else None),
        max_workers=(int(args.max_workers) if args.max_workers is not None else None),
        max_draws=(int(args.max_draws) if args.max_draws is not None else None),
    )

    print("BestCP:", bestCP)
    print("F(S) ≈", bestF)


if __name__ == "__main__":
    main()
