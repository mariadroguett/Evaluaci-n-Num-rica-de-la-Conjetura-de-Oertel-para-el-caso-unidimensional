#!/usr/bin/env python3
# ===========================================================
# run_ortel_parallel.py — versión seedless (NPZ only)
# Ejecuta varias réplicas en paralelo llamando a main_ortel.py
# ===========================================================
import os
import sys
import uuid
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ===== Configuración general =====
# Aquí cada clave es "n_per_z": puntos por fibra
POINTS_AND_REPS = {
    5:  50,
    8:  50,
    11: 25,
    15: 13,
    20: 13,
    # Para el run grande puedes usar:
    # 5:  100,
    # 8:  100,
    # 11: 50,
    # 15: 25,
    # 20: 25,
}

N         = 30000
N_CP      = 200
N_HIP     = 500
D         = 2
Z_VALS    = [0, 1, 2]        # tres fibras
F_THRESH  = 0.18
TARGET_MB = 64.0

# Paralelismo externo (procesos independientes)
# 👇 CAMBIO IMPORTANTE: usar los CPUs que SLURM asigna (ej: 16)
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))

# ===== Paths =====
PROJECT_DIR = Path(__file__).resolve().parent
MAIN        = PROJECT_DIR / "main_ortel.py"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR    = PROJECT_DIR / "logs_runs"
for p in [RESULTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def py_exe() -> str:
    """Devuelve el ejecutable de Python actual."""
    return sys.executable or "python"


def job_cmd(n_per_z: int) -> list[str]:
    """
    Construye el comando que ejecuta main_ortel.py con parámetros fijos.
    Aquí n_per_z es el nº de puntos por fibra.
    """
    return [
        py_exe(), "-X", "utf8", str(MAIN),
        "--d", str(D),
        "--z_vals", *[str(z) for z in Z_VALS],
        "--n_per_z", str(n_per_z),
        "--N", str(N),
        "--N_cp", str(N_CP),
        "--N_hip", str(N_HIP),
        "--f_threshold", str(F_THRESH),
        "--target_mb", str(TARGET_MB),
        "--results_root", str(RESULTS_DIR),
    ]


def run_one(n_per_z: int) -> tuple[bool, int, str]:
    """Ejecuta una réplica individual y guarda sus logs."""
    rid = uuid.uuid4().hex[:8]
    log_out = LOGS_DIR / f"run_np{n_per_z}_{rid}.out"
    log_err = LOGS_DIR / f"run_np{n_per_z}_{rid}.err"

    cmd = job_cmd(n_per_z)
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Guardamos logs por réplica (útil para revisar errores)
    log_out.write_text(proc.stdout or "", encoding="utf-8")
    log_err.write_text(proc.stderr or "", encoding="utf-8")

    ok = (proc.returncode == 0)
    if not ok:
        sys.stderr.write(f"[ERR] n_per_z={n_per_z} rid={rid} rc={proc.returncode}\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr.strip()[:1500] + "\n")
    return ok, n_per_z, rid


# ===== Main loop =====
def main() -> int:
    if not MAIN.exists():
        sys.stderr.write(f"[FATAL] No encuentro main_ortel.py en {MAIN}\n")
        return 2

    print(f"=== Lanzando experimentos seedless con {NUM_WORKERS} workers ===")
    print(f"Z={Z_VALS} | D={D} | N={N} | N_cp={N_CP} | N_hip={N_HIP} | thr={F_THRESH}")
    print(f"POINTS_AND_REPS={POINTS_AND_REPS}")

    for n_per_z, reps in POINTS_AND_REPS.items():
        print(f"\n>>> n_per_z={n_per_z} — {reps} réplicas")
        ok_cnt = err_cnt = 0

        if reps <= 0:
            print(f"[SKIP] n_per_z={n_per_z} sin réplicas")
            continue

        # Pool de procesos paralelos
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = [ex.submit(run_one, n_per_z) for _ in range(reps)]
            for i, fut in enumerate(as_completed(futs), 1):
                ok, npz, rid = fut.result()
                ok_cnt += int(ok)
                err_cnt += int(not ok)
                if (i % 10 == 0) or (not ok):
                    print(f"[PROG] n_per_z={n_per_z}: {i}/{reps} (ok={ok_cnt}, err={err_cnt})")

        print(f"[DONE] n_per_z={n_per_z}: ok={ok_cnt}, err={err_cnt}")

    print("\n=== TODO COMPLETADO ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
