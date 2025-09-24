#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any, Optional

# Limitar subprocesos de BLAS/OpenMP para evitar oversubscription.
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
})

from concurrent.futures import ProcessPoolExecutor, as_completed
#from typing import List, Dict, Any, Optional
#import argparse, json

import numpy as np
from convex_hull import generate_convex_hull, random_vertices_by_fiber, load_fibers_from_dir
from ortel import ortel

# --- setup fijo ---
D = 2
Z_VALS = [0, 1]
N_PER_Z = 5  # número de puntos por fibra
PARAMS = dict(
    N_cp=50,
    N_hip=100,
    N=5_000_000,
    tol=1e-9,
    batch=5_000,
    guided=False,
    defer_rejection=True,
)

def one_run(seed: int, fibers_root: Optional[str] = None, reuse_fibers: bool = True, cp_out_root: Optional[str] = None) -> Dict[str, Any]:
    """Ejecuta una corrida completa (vértices -> hull -> ORTEL) para una semilla."""
    # 1) Vértices: reutilizar o generar y guardar por semilla
    verts = None
    fiber_dir = None
    if fibers_root is not None:
        fiber_dir = os.path.join(fibers_root, f"seed_{int(seed)}")
        if reuse_fibers and os.path.isdir(fiber_dir):
            try:
                verts = load_fibers_from_dir(D, fiber_dir)
                print(f"seed={seed}: loaded fibers from {fiber_dir}")
            except Exception as e:
                print(f"seed={seed}: failed to load fibers, regenerating: {e}")
                verts = None

    if verts is None:
        # generate and optionally save
        if fiber_dir is None and fibers_root is not None:
            fiber_dir = os.path.join(fibers_root, f"seed_{int(seed)}")
        verts = random_vertices_by_fiber(Z_VALS, d=D, n_per_z=N_PER_Z, seed=seed, save_fibers_dir=fiber_dir)
    
    triangle = [[0,0,0],[0,0,1],[0,1/2,1],[1,0,0],[1,0,1],[1,1/2,1]]  # triángulo con vértices en z=0 y z=1

    # 2) Hull
    A, b = generate_convex_hull(triangle)

    # 2) ORTEL
    save_dir = None
    if cp_out_root is not None:
        save_dir = os.path.join(cp_out_root, f"seed_{int(seed)}")
    bestCP, bestF = ortel(A, b, D, z_vals=Z_VALS, seed=seed, save_dir=save_dir, **PARAMS)
    return {"seed": int(seed), "F": float(bestF), "bestCP": bestCP.tolist()}


def run_cpu_tasks_in_parallel(seeds: List[int], max_workers: Optional[int] = None, fibers_root: Optional[str] = None, reuse_fibers: bool = True, cp_out_root: Optional[str] = None):
    if not seeds:
        return []
    if max_workers is None:
        max_workers = min(len(seeds), os.cpu_count() or 1)

    results = []
    total = len(seeds)
    print(f"Lanzando {total} corridas con hasta {max_workers} procesos...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        indexed = list(enumerate(seeds, start=1))
        futures = {}
        for idx, s in indexed:
            print(f"[{idx}/{total}] En cola corrida seed={s}")
            fut = executor.submit(one_run, s, fibers_root, reuse_fibers, cp_out_root)
            futures[fut] = (idx, s)

        done = 0
        for fut in as_completed(futures):
            idx, seed = futures[fut]
            done += 1
            try:
                res = fut.result()
                results.append(res)
                print(f"[{done}/{total}] OK seed={seed} F={res['F']:.6f}")
            except Exception as e:
                print(f"[{done}/{total}] FAIL seed={seed}: {e}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experimentos ORTEL en paralelo (multi-proceso)")
    parser.add_argument("--seeds", type=str, default="105,106,107,108,109gi", help="Lista de semillas separadas por coma")
    parser.add_argument("--workers", type=int, default=5, help="Nº de procesos en paralelo")
    parser.add_argument("--outdir", type=str, default="results", help="Carpeta para guardar JSONs y artefactos")
    parser.add_argument("--fibers-dir", type=str, default=None, help="Carpeta base para guardar/cargar fibras por semilla (dentro de outdir si es relativa)")
    parser.add_argument("--reuse-fibers", action="store_true", help="Si existe la carpeta de fibras por semilla, la reutiliza")
    parser.add_argument("--cp-out", type=str, default=None, help="Carpeta base para CSVs de CPs aceptados (por semilla)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Resolver rutas
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    fibers_root = None
    if args.fibers_dir:
        fibers_root = args.fibers_dir
        if not os.path.isabs(fibers_root):
            fibers_root = os.path.join(outdir, fibers_root)
        os.makedirs(fibers_root, exist_ok=True)
    cp_out_root = None
    if args.cp_out:
        cp_out_root = args.cp_out
        if not os.path.isabs(cp_out_root):
            cp_out_root = os.path.join(outdir, cp_out_root)
        os.makedirs(cp_out_root, exist_ok=True)

    results = run_cpu_tasks_in_parallel(
        seeds,
        max_workers=args.workers,
        fibers_root=fibers_root,
        reuse_fibers=args.reuse_fibers,
        cp_out_root=cp_out_root,
    )

    # Guardar cada corrida en JSON consolidado
    for r in results:
        out = {
            "mode": "ortel_parallel",
            "d": D,
            "z_vals": Z_VALS,
            "n_per_z": N_PER_Z,
            **PARAMS,
            "seed": r["seed"],
            "F": r["F"],
            "bestCP": r["bestCP"],
        }
        path = os.path.join(outdir, f"seed_{r['seed']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"wrote {path}")

    print(f"Completed {len(results)} runs.")
