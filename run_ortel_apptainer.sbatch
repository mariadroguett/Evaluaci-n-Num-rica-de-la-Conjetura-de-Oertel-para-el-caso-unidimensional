#!/bin/bash
#SBATCH -J ortel_run
#SBATCH -p ngen-ko
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o logs_slurm/%x-%j.out
#SBATCH -e logs_slurm/%x-%j.err

# Mejor: sin -u, o usa defaults al leer variables
set -eo pipefail

mkdir -p logs_slurm results tmp_csv results/hulls

# Evitar oversubscription
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

ROOT="$PWD"
WORK="/workspace"
SIF="$ROOT/ortel.sif"

echo "=== Nodo: $(hostname) ==="
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}   Mem: ${SLURM_MEM_PER_NODE:-?}   Time: ${SLURM_TIMELIMIT:-?}"
echo "Contenedor: $SIF"
echo "========================="

# Opcional: limitar workers internos
# export MAX_WORKERS=${SLURM_CPUS_PER_TASK:-4}

if command -v apptainer >/dev/null 2>&1; then
  APPT=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPT=singularity
else
  echo "[FATAL] No encuentro apptainer/singularity en el nodo" >&2
  exit 127
fi

$APPT exec --bind "$ROOT":"$WORK" "$SIF" \
  python3 -u /workspace/run_ortel_parallel.py

echo "=== FIN ==="
