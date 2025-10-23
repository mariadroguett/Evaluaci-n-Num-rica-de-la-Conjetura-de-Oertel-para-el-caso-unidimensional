#!/bin/bash
#SBATCH -J ortel_smoke
#SBATCH -p ngen-ko
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -o logs_slurm/%x-%j.out
#SBATCH -e logs_slurm/%x-%j.err

set -euo pipefail
set -x

mkdir -p logs_slurm results results/hulls tmp_csv

# Rutas absolutas (evita bind roto)
ROOT="$(pwd -P)"
SIF="$(readlink -f ortel.sif)"
WORK="/workspace"

echo "== who/host/pwd =="
whoami; hostname; pwd

echo "== apptainer =="
module load apptainer 2>/dev/null || true
command -v apptainer || true
apptainer --version || true

echo "== archivos en submit dir =="
ls -lh "$SIF" || true
ls -lh main_ortel.py run_ortel_parallel.py ortel.sif || true

echo "== probar python dentro del contenedor =="
# intenta python3 y si no, python
apptainer exec --bind "$ROOT":"$WORK" "$SIF" bash -lc 'python3 -V || python -V'

echo "== listar /workspace dentro del contenedor =="
apptainer exec --bind "$ROOT":"$WORK" "$SIF" bash -lc 'ls -lh /workspace | head -n 50; ls -lh /workspace/main_ortel.py || true'

echo "== SMOKE: 1 corrida mini =="
apptainer exec --bind "$ROOT":"$WORK" "$SIF" \
  bash -lc 'python3 /workspace/main_ortel.py \
    --seed 1 --n_point 5 --d 2 --N 300000 --N_cp 100 --N_hip 1000 \
    --f_threshold 0.18 \
    --save_hull_dir /workspace/results/hulls \
    --out /workspace/tmp_csv/smoke_seed1.csv || \
    python /workspace/main_ortel.py \
    --seed 1 --n_point 5 --d 2 --N 20000 --N_cp 10 --N_hip 50 \
    --f_threshold 0.18 \
    --save_hull_dir /workspace/results/hulls \
    --out /workspace/tmp_csv/smoke_seed1.csv'

echo "== listo =="

