#!/usr/bin/env python3
import os, sys, subprocess, csv
from pathlib import Path

# ---- config mínima ----
REPS        = 100           # 100 seeds por n_point
N_POINTS    = [5]           # puedes poner [5, 8] si quieres dos tandas
N           = 300000
N_CP        = 100
N_HIP       = 1000
N_PER_Z     = 5
D           = 2
F_THRESHOLD = 0.18

# rutas
PROJECT_DIR = Path(__file__).resolve().parent
MAIN        = PROJECT_DIR / "main_ortel.py"
RESULTS_DIR = PROJECT_DIR / "results"
HULLS_DIR   = RESULTS_DIR / "hulls"
CSV_GLOBAL  = RESULTS_DIR / "experiments.csv"

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HULLS_DIR.mkdir(parents=True, exist_ok=True)

def csv_has_row(csv_path: Path, seed: int, n_point: int) -> bool:
    """
    Evita duplicados si relanzas: chequea si ya existe una fila con ese n_point y seed.
    (Tu main escribe columnas: ["n_point","F","bestcp"], no incluye seed.
     Si quieres chequeo perfecto, agrega seed a tu main. Por ahora, solo prevenimos repetir
     si el archivo existe y ya corrimos todas las seeds: dejamos esta función por si luego amplías.)
    """
    # Con el formato actual del CSV no podemos saber el seed, así que devolvemos False siempre.
    # Si decides agregar 'seed' a tu main, implementa el chequeo acá.
    return False

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
    print(f"[RUN] seed={seed} n_point={n_point}")
    return subprocess.call(cmd, cwd=PROJECT_DIR)

def main():
    # UTF-8 siempre
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if not MAIN.exists():
        print(f"[FATAL] No encuentro main_ortel.py en {MAIN}", file=sys.stderr)
        sys.exit(2)

    ensure_dirs()

    total = len(N_POINTS) * REPS
    done_ok = done_err = 0

    # Cabecera del CSV si no existe aún
    if not CSV_GLOBAL.exists():
        with CSV_GLOBAL.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n_point", "F", "bestcp"])

    for n_point in N_POINTS:
        for seed in range(1, REPS + 1):
            # si implementas seed en el CSV, aquí podrías saltar si ya existe
            if csv_has_row(CSV_GLOBAL, seed, n_point):
                print(f"[SKIP] ya registrado seed={seed} n_point={n_point}")
                continue

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

