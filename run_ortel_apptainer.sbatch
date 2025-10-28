#!/bin/bash
#SBATCH -J ortel_run
#SBATCH -p ngen-ko
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH -o logs_slurm/%x-%j.out
#SBATCH -e logs_slurm/%x-%j.err

set -euo pipefail

mkdir -p logs_slurm results results/hulls results/tmp

# BLAS/OpenMP dentro del contenedor = núcleos asignados
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

ROOT="$PWD"
SIF="$ROOT/ortel.sif"
WORK="/workspace"

echo "=== Nodo: $(hostname) ==="
echo "CPUs: ${SLURM_CPUS_PER_TASK}   Mem: ${SLURM_MEM_PER_NODE:-?}   Time: ${SLURM_TIMELIMIT:-?}"
echo "Contenedor: $SIF"
echo "========================="

apptainer exec --bind "$ROOT":"$WORK" "$SIF" \
  python3 "$WORK/run_ortel_parallel.py"
