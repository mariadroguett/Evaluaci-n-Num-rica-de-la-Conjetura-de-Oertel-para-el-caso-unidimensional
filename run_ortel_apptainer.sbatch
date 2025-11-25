#!/bin/bash
#SBATCH -J ortel_parallel
#SBATCH -p ngen-ko
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 72:00:00
#SBATCH -o logs_slurm/%x-%j.out
#SBATCH -e logs_slurm/%x-%j.err

set -euo pipefail

echo "=== Nodo: $(hostname) ==="
echo "CPUs asignadas: ${SLURM_CPUS_PER_TASK}"
echo "Fecha: $(date)"
echo "=========================="

# Ir al repo
cd /mnt/beegfs/home/mdroguett/Memoria/Evaluaci-n-Num-rica-de-la-Conjetura-de-Oertel-para-el-caso-unidimensional

# Crear carpetas necesarias
mkdir -p logs_slurm results results/hulls results/hulls_obs logs_runs tmp_csv

# Desactivar paralelismo interno de BLAS/OpenMP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Activar entorno mamba

#mamba activate experimento_parallel

echo "=== Lanzando run_ortel_parallel.py ==="
time python3 run_ortel_parallel.py
echo "=== FIN OK ==="
