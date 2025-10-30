#!/usr/bin/env python3
import os, sys, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Config =====
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
Z_VALS      = [0, 1]
N_PER_Z     = 5
F_THRESHOLD = 0.18
TARGET_MB   = 64.0  # se pasa al main (aunque no sea usado dentro de ortel.py)

# Workers (detecta SLURM si aplica)
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

# ===== Paths =====
PROJECT_DIR = Path(__file__).resolve().parent
MAIN        = PROJECT_DIR / "main_ortel.py"
RESULTS_DIR = PROJECT_DIR / "results"
TMP_DIR     = PROJECT_DIR / "tmp_csv"
HULLS_DIR   = RESULTS_DIR / "hulls"
HULLS_OBS   = RESULTS_DIR / "hulls_obs"

for p in [RESULTS_DIR, TMP_DIR, HULLS_DIR, HULLS_OBS]:
    p.mkdir(parents=True, exist_ok=True)


def py_exe():
    return sys.executable or "python"


def job_cmd(seed: int, n_point: int) -> list[str]:
    # cada seed escribe un csv temporal propio
    out_tmp = TMP_DIR / f"exp_{n_point}__seed_{seed}.csv"
    return [
        py_exe(), "-X", "utf8", str(MAIN),
        "--seed", str(seed),
        "--d", str(D),
        "--z_vals", *[str(z) for z in Z_VALS],
        "--n_per_z", str(N_PER_Z),
        "--n_point", str(n_point),
        "--N", str(N),
        "--N_cp", str(N_CP),
        "--N_hip", str(N_HIP),
        "--f_threshold", str(F_THRESHOLD),
        "--target_mb", str(TARGET_MB),
        "--save_hull_dir", str(HULLS_DIR),
        "--save_hull_obs_dir", str(HULLS_OBS),
        "--out", str(out_tmp),
    ]


def run_one(seed: int, n_point: int) -> tuple[bool, int, int]:
    out_tmp = TMP_DIR / f"exp_{n_point}__seed_{seed}.csv"
    if out_tmp.exists():
        print(f"[SKIP] n_point={n_point} seed={seed} (ya existe {out_tmp.name})")
        return True, seed, n_point

    cmd = job_cmd(seed, n_point)
    print(f"[RUN]  n_point={n_point} seed={seed}")
    proc = subprocess.run(
        cmd, cwd=PROJECT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", shell=False
    )
    ok = (proc.returncode == 0)
    if not ok:
        # Log mínimo al stderr del runner
        sys.stderr.write(f"[ERR] seed={seed} n_point={n_point} rc={proc.returncode}\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr.strip() + "\n")
    return ok, seed, n_point


def merge_by_npoint(n_point: int):
    """Funde tmp_csv/exp_<n_point>__seed_*.csv -> results/exp_<n_point>.csv"""
    target = RESULTS_DIR / f"exp_{n_point}.csv"
    # recolecta todos los temporales de ese n_point
    files = sorted(TMP_DIR.glob(f"exp_{n_point}__seed_*.csv"))
    if not files:
        print(f"[WARN] no hay temporales para n_point={n_point}")
        return

    # escribimos header 1 vez
    header_written = False
    with target.open("w", encoding="utf-8", newline="") as out:
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                continue
            if not header_written:
                out.write(lines[0])   # header
                header_written = True
            out.writelines(lines[1:])  # datos

    print(f"[MERGE] {len(files)} archivos → {target}")


def main():
    if not MAIN.exists():
        sys.stderr.write(f"[FATAL] No encuentro main_ortel.py en {MAIN}\n")
        return 2

    # Para cada n_point, armamos su lista de seeds y corremos en paralelo
    for n_point, reps in POINTS_AND_REPS.items():
        seeds = list(range(1, reps + 1))
        print(f"\n=== Lanzando n_point={n_point} con {reps} seeds usando {NUM_WORKERS} workers ===")
        ok_cnt = err_cnt = 0
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = {ex.submit(run_one, s, n_point): s for s in seeds}
            for i, fut in enumerate(as_completed(futs), 1):
                ok, seed, _np = fut.result()
                if ok:
                    ok_cnt += 1
                else:
                    err_cnt += 1
                if i % 10 == 0 or not ok:
                    print(f"[PROG] n_point={n_point}: {i}/{reps} (ok={ok_cnt}, err={err_cnt})")

        # merge por n_point
        merge_by_npoint(n_point)

    print("\n=== TODO COMPLETADO ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
