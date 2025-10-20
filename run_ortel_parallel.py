#!/usr/bin/env python3
import os, sys, subprocess
from pathlib import Path

# ---- config mínima ----
REPS        = 1           # 100 seeds por n_point
N_POINTS    = [5, 8]
N           = 20_000
N_CP        = 20
N_HIP       = 500
N_PER_Z     = 5
D           = 2
F_THRESHOLD = 0.18

# rutas
PROJECT_DIR = Path(__file__).resolve().parent
MAIN        = PROJECT_DIR / "main_ortel.py"
RESULTS_DIR = PROJECT_DIR / "results"
HULLS_DIR   = RESULTS_DIR / "hulls"
CSV_GLOBAL  = RESULTS_DIR / "experiments.csv"

def run_one(seed: int, n_point: int) -> int:
    cmd = [
        sys.executable or "python", "-X", "utf8", str(MAIN),
        "--seed", str(seed),
        "--n_point", str(n_point),
        "--N", str(N),
        "--N_cp", str(N_CP),
        "--N_hip", str(N_HIP),
        "--n_per_z", str(N_PER_Z),
        "--d", str(D),
        "--f_threshold", str(F_THRESHOLD),
        "--save_hull_dir", str(HULLS_DIR),
        "--out", str(CSV_GLOBAL),
    ]
    # imprime algo breve para seguir el avance
    print(f"[RUN] seed={seed} n_point={n_point}")
    return subprocess.call(cmd, cwd=PROJECT_DIR)

def main():
    if not MAIN.exists():
        print(f"[FATAL] No encuentro main_ortel.py en {MAIN}", file=sys.stderr)
        sys.exit(2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HULLS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(N_POINTS) * REPS
    done_ok = done_err = 0

    for n_point in N_POINTS:
        for seed in range(1, REPS + 1):
            rc = run_one(seed, n_point)
            if rc == 0:
                done_ok += 1
            else:
                done_err += 1
                print(f"[ERR] seed={seed} n_point={n_point} (rc={rc})", file=sys.stderr)

    print(f"\n[FIN] OK={done_ok}  ERR={done_err}")
    print(f"[OUT] CSV global: {CSV_GLOBAL}")
    print(f"[OUT] Hulls (si F<{F_THRESHOLD}): {HULLS_DIR}")

if __name__ == "__main__":
    main()
