#!/bin/bash
#SBATCH -J ortel_run
#SBATCH -p ngen-ko                 # partición correcta según tu sinfo
#SBATCH -N 1
#SBATCH --cpus-per-task=16         # núcleos que verá tu Python
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o logs_slurm/%x-%j.out
#SBATCH -e logs_slurm/%x-%j.err

set -eo pipefail
set -x

# --- Paths ---
ROOT="$PWD"
WORK="/workspace"
SIF="$ROOT/ortel.sif"

mkdir -p logs_slurm results tmp_csv results/hulls

# --- Encontrar apptainer/singularity ---
APPT=$(command -v apptainer || command -v singularity || true)
if [[ -z "${APPT}" ]]; then
  echo "[FATAL] No se encontró 'apptainer' ni 'singularity' en el nodo."
  exit 127
fi

# --- Limitar BLAS/OpenMP (y pasar núcleos al runner) ---
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_MAX_THREADS="$SLURM_CPUS_PER_TASK"
export SLURM_CPUS_PER_TASK   # para que lo lea tu run_ortel_parallel.py

echo "=== Nodo: $(hostname) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem: $SLURM_MEM_PER_NODE   Time: $SLURM_TIMELIMIT"
echo "Contenedor: $SIF"
echo "========================="

# --- sanity quick check dentro del contenedor ---
"${APPT}" exec --bind "$ROOT":"$WORK" "$SIF" python3 -V

# --- correr tu runner paralelo (usa SLURM_CPUS_PER_TASK como workers) ---
"${APPT}" exec --bind "$ROOT":"$WORK" "$SIF" \
  python3 "$WORK/run_ortel_parallel.py"

echo "=== FIN OK ==="
