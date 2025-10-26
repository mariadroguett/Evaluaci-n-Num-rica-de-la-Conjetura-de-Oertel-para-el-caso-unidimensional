#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Config =====
# 100×(n_point=5) y 100×(n_point=8)  → ajusta como quieras
REPS        = 100
N_POINTS    = [5, 8]

# Coste por corrida: ajusta según cluster/PC
N           = 300_000
N_CP        = 100
N_HIP       = 1_000
D           = 2
N_PER_Z     = 5
Z_VALS      = [0, 1]
F_THRESHOLD = 0.18

# Paralelismo
DEF_WORKERS = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "4")))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", DEF_WORKERS))

# Rutas
PROJECT_DIR = Path(__file__).resolve().parent
MAIN        = PROJECT_DIR / "main_ortel.py"
RESULTS_DIR = PROJECT_DIR / "results"
HULLS_DIR   = RESULTS_DIR / "hulls"
TMP_CSV_DIR = PROJECT_DIR / "tmp_csv"
MERGED_CSV  = RESULTS_DIR / "experiments.csv"

# Evitar oversubscription de BLAS/OpenMP (mejor 1 por proceso)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HULLS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_CSV_DIR.mkdir(parents=True, exist_ok=True)


def py_exe():
    return sys.executable or "python"


def out_csv_for(seed: int, n_point: int) -> Path:
    return TMP_CSV_DIR / f"seed_{seed}_npoint_{n_point}.csv"


def job_cmd(seed: int, n_point: int):
    out_csv = out_csv_for(seed, n_point)
    cmd = [
        py_exe(), "-X", "utf8", str(MAIN),
        "--seed", str(seed),
        "--n_point", str(n_point),
        "--N", str(N),
        "--N_cp", str(N_CP),
        "--N_hip", str(N_HIP),
        "--n_per_z", str(N_PER_Z),
        "--d", str(D),
        "--f_threshold", str(F_THRESHOLD),
        "--save_hull_dir", str(HULLS_DIR),
        "--out", str(out_csv),
    ]
    # z_vals explícitos (dos fibras por defecto)
    for z in Z_VALS:
        cmd += ["--z_vals", str(z)]
    return cmd


def run_one(seed: int, n_point: int):
    out_csv = out_csv_for(seed, n_point)
    # Reanudable: si ya existe y tiene contenido, saltar
    if out_csv.exists() and out_csv.stat().st_size > 0:
        print(f"[SKIP] seed={seed} n_point={n_point} (ya está {out_csv.name})")
        return True, seed, n_point

    cmd = job_cmd(seed, n_point)
    print(f"[RUN] seed={seed} n_point={n_point}")
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False
    )
    if proc.stdout:
        # Log compacto por si corres en Slurm (aparecerá en .out)
        sys.stdout.write(proc.stdout.rstrip() + "\n")
        sys.stdout.flush()
    ok = (proc.returncode == 0)
    if not ok:
        sys.stderr.write(f"[ERR] seed={seed} n_point={n_point} (rc={proc.returncode})\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr.rstrip() + "\n")
        sys.stderr.flush()
    return ok, seed, n_point


def merge_csvs():
    csvs = sorted(TMP_CSV_DIR.glob("*.csv"))
    if not csvs:
        print("[WARN] No hay CSVs en tmp_csv/ para fusionar.")
        return

    MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)

    # merge con encabezado único; si repites el runner, no duplica
    header = None
    lines_seen = set()
    with MERGED_CSV.open("w", encoding="utf-8", newline="") as out:
        for fp in csvs:
            with fp.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
                out.write(header)
                start = 1
            else:
                # si el header coincide, saltamos la 1ª línea
                start = 1 if lines[0] == header else 0

            for ln in lines[start:]:
                if ln in lines_seen:
                    continue
                lines_seen.add(ln)
                out.write(ln)


def main():
    if not MAIN.exists():
        print(f"[FATAL] No encuentro main_ortel.py en {MAIN}", file=sys.stderr)
        sys.exit(2)

    ensure_dirs()

    jobs = [(seed, npt) for npt in N_POINTS for seed in range(1, REPS + 1)]
    total = len(jobs)
    print(f"[INFO] Lanzando {total} corridas en paralelo (max_workers={MAX_WORKERS})…")

    done_ok = done_err = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, s, n) for (s, n) in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            ok, seed, n_point = fut.result()
            if ok:
                done_ok += 1
                print(f"[{i}/{total}] OK seed={seed} n_point={n_point}")
            else:
                done_err += 1
                print(f"[{i}/{total}] ERR seed={seed} n_point={n_point}", file=sys.stderr)

    print(f"[INFO] Finalizado: OK={done_ok}  ERR={done_err}. Haciendo merge…")
    merge_csvs()
    print(f"[INFO] CSV global: {MERGED_CSV}")
    print(f"[INFO] Hulls (si F<{F_THRESHOLD}): {HULLS_DIR}")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("Interrumpido por usuario", file=sys.stderr)
        sys.exit(130)
