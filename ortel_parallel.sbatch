#!/usr/bin/env bash
#SBATCH --job-name=ortel_parallel
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=ngen-ko     

set -euo pipefail

# ====== DIAGNÓSTICO Y ENTORNO ======
# Evita oversubscription de BLAS/OpenMP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Activa conda/mamba en modo no interactivo
source ~/.bashrc || true
eval "$(conda shell.bash hook)"
conda activate experimento_parallel

# Logs + directorios de salida
OUTDIR=${OUTDIR:-"results"}
FIBERS_DIR=${FIBERS_DIR:-"fibras"}      # relativo a OUTDIR
CP_OUT=${CP_OUT:-"cp_aceptados"}        # relativo a OUTDIR
SEEDS=${SEEDS:-"100,101,102,103,104"}

# Usa todos los CPUs asignados por Slurm salvo que especifiques WORKERS
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}

mkdir -p logs "$OUTDIR" "$OUTDIR/$FIBERS_DIR" "$OUTDIR/$CP_OUT"

echo "[SLURM] Host: $(hostname)"
echo "[SLURM] JobID: ${SLURM_JOB_ID:-NA}"
echo "[SLURM] CWD  : $(pwd)"
echo "[SLURM] Python: $(which python3)"
python3 --version

# ====== EJECUCIÓN ======

srun --ntasks=1 --cpus-per-task="${WORKERS}" \
  python3 exp_ortel_parallel.py \
    --seeds "${SEEDS}" \
    --workers "${WORKERS}" \
    --outdir "${OUTDIR}" \
    --fibers-dir "${FIBERS_DIR}" \
    --reuse-fibers \
    --cp-out "${CP_OUT}"

echo "[SLURM] Done."

