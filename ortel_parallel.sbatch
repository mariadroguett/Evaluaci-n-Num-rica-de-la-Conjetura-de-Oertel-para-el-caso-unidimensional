#!/bin/bash
#SBATCH --job-name=ortel_parallel
#SBATCH --output=logs/ortel_parallel_%j.out
#SBATCH --error=logs/ortel_parallel_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
##SBATCH --partition=cpu
##SBATCH --account=your_account

set -euo pipefail

# Ajusta estas variables según necesites
SEEDS=${SEEDS:-"100,101,102,103,104"}
OUTDIR=${OUTDIR:-"results"}
FIBERS_DIR=${FIBERS_DIR:-"fibras"}   # relativo a OUTDIR
CP_OUT=${CP_OUT:-"cp_aceptados"}     # relativo a OUTDIR
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}

# module load python/3.10
# source ~/.venvs/ortel/bin/activate

# Limitar hilos BLAS/OpenMP (evita sobre-suscripción)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

echo "[SLURM] Host: $(hostname) | JobID: ${SLURM_JOB_ID:-NA}"
echo "[SLURM] CWD : $(pwd)"
echo "[SLURM] Seeds=${SEEDS} | Workers=${WORKERS} | Outdir=${OUTDIR}"

mkdir -p logs "${OUTDIR}"

python3 exp_ortel_parallel.py \
  --seeds "${SEEDS}" \
  --workers "${WORKERS}" \
  --outdir "${OUTDIR}" \
  --fibers-dir "${FIBERS_DIR}" \
  --reuse-fibers \
  --cp-out "${CP_OUT}"

echo "[SLURM] Done."

