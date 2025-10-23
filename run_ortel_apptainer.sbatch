#!/bin/bash
#SBATCH -J ortel_run
#SBATCH -p ngen-ko                # partición del cluster
#SBATCH -N 1
#SBATCH --cpus-per-task=4         # núcleos por tarea
#SBATCH --mem=6G                  # memoria total
#SBATCH -t 06:00:00               # tiempo máximo
#SBATCH -o logs_slurm/%x-%j.out   # salida estándar
#SBATCH -e logs_slurm/%x-%j.err   # errores

set -euo pipefail
set -x  # imprime los comandos para debug

# Crear carpetas si no existen
mkdir -p logs_slurm results tmp_csv results/hulls

# Limitar paralelismo de BLAS/OpenMP dentro del contenedor
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

# Directorio actual y contenedor
ROOT="$PWD"
WORK="/workspace"
SIF="$ROOT/ortel.sif"

echo "=== Ejecutando en nodo $(hostname) ==="
echo "Directorio de trabajo: $ROOT"
echo "Usando contenedor: $SIF"
echo "==============================="

# Ejecutar el experimento dentro del contenedor
apptainer exec --bind "$ROOT":"$WORK" "$SIF" \
  python3 /workspace/run_ortel_parallel.py

echo "=== Ejecución completada ==="
