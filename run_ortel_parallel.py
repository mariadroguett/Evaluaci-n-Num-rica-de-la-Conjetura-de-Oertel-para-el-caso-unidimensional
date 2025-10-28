# run_ortel_parallel.py
#!/usr/bin/env python3
import os, sys, csv, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Setup de experimentos =====
POINTS_AND_REPS = {
    5:  100,
    8:  100,
    11: 50,
    15: 25,
    20: 25,
}

N           = 300_000
N_CP        = 100
N_HIP       = 1_000
D           = 2
N_PER_Z     = 5
Z_VALS      = [0, 1, 2]
F_THRESHOLD = 0.18
TARGET_MB   = 64.0

# Workers: usa SLURM_CPUS_PER_TASK si existe
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

# Rutas
PROJECT_DIR = Path(__file__).resolve().parent
MAIN        = PROJECT_DIR / "main_ortel.py"
RESULTS_DIR = PROJECT_DIR / "results"
TMP_DIR     = RESULTS_DIR / "tmp"
HULLS_DIR   = RESULTS_DIR / "hulls"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
HULLS_DIR.mkdir(parents=True, exist_ok=True)

def py_exe():
    return sys.executable or "python"

def tmp_csv(n_point: int, seed: int) -> Path:
    d = TMP_DIR / f"exp_{n_point}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"seed_{seed}.csv"

def final_csv(n_point: int) -> Path:
    return RESULTS_DIR / f"exp_{n_point}.csv"

def job_cmd(seed: int, n_point: int):
    out = tmp_csv(n_point, seed)
    return [
        py_exe(), "-X", "utf8", str(MAIN),
        "--seed", str(seed),
        "--n_point", str(n_point),
        "--N", str(N),
        "--N_cp", str(N_CP),
        "--N_hip", str(N_HIP),
        "--n_per_z", str(N_PER_Z),
        "--d", str(D),
        "--z_vals", *map(str, Z_VALS),
        "--f_threshold", str(F_THRESHOLD),
        "--target_mb", str(TARGET_MB),
        "--save_hull_dir", str(HULLS_DIR),
        "--out", str(out),
    ]

def run_one(seed: int, n_point: int):
    out = tmp_csv(n_point, seed)
    if out.exists() and out.stat().st_size > 0:
        print(f"[SKIP] {out}")
        return 0, seed, n_point

    cmd = job_cmd(seed, n_point)
    print(f"[RUN] n_point={n_point} seed={seed}")
    rc = subprocess.call(cmd, cwd=PROJECT_DIR)
    return rc, seed, n_point

def merge_per_npoint(n_point: int):
    dst = final_csv(n_point)
    # juntar todos los tmp de ese n_point
    parts = sorted((TMP_DIR / f"exp_{n_point}").glob("seed_*.csv"))
    if not parts:
        print(f"[WARN] no hay partes para n_point={n_point}")
        return
    header_written = False
    with dst.open("w", encoding="utf-8", newline="") as out:
        wdst = None
        for fp in parts:
            with fp.open("r", encoding="utf-8", newline="") as f:
                rdr = csv.reader(f)
                rows = list(rdr)
            if not rows:
                continue
            if not header_written:
                out.write(",".join(rows[0]) + "\n")
                header_written = True
            for row in rows[1:]:
                out.write(",".join(map(str, row)) + "\n")

def main():
    if not MAIN.exists():
        print(f"[FATAL] No existe {MAIN}", file=sys.stderr)
        return 2

    jobs = []
    for n_point, reps in POINTS_AND_REPS.items():
        for seed in range(1, reps + 1):
            # si ya está en el CSV final, no re-ejecutamos
            dst = final_csv(n_point)
            if dst.exists():
                with dst.open("r", encoding="utf-8") as f:
                    if any(line.startswith(f"{seed},{n_point}") for line in f):
                        continue
            jobs.append((seed, n_point))

    print(f"=== Lanzando {len(jobs)} jobs con {NUM_WORKERS} workers ===")
    ok = err = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [ex.submit(run_one, s, npt) for (s, npt) in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rc, seed, npt = fut.result()
            if rc == 0:
                ok += 1
            else:
                err += 1
                print(f"[ERR] seed={seed} n_point={npt} rc={rc}", file=sys.stderr)
            if i % 20 == 0:
                print(f"[PROGRESS] {i}/{len(jobs)}")

    # Merge por n_point
    for n_point in POINTS_AND_REPS.keys():
        merge_per_npoint(n_point)

    print(f"=== FIN: OK={ok} ERR={err} ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
