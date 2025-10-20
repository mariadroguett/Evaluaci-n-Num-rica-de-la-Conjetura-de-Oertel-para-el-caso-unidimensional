#!/bin/bash
#SBATCH -J ortel
#SBATCH -p normal            # cambia por tu partición
#SBATCH -N 1
#SBATCH -c 8                 # 8 cores; ajusta
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o logs_slurm/%x-%j.out
#SBATCH -e logs_slurm/%x-%j.err

set -euo pipefail
module load apptainer 2>/dev/null || true

mkdir -p logs_slurm results results/hulls tmp_csv

# Recomendado: limitar BLAS dentro del job también
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Ejecuta tu runner paralelo (usa -c para cores del nodo)
apptainer exec --bind "$PWD":/workspace ortel.sif \
  python /workspace/run_ortel_parallel.py
