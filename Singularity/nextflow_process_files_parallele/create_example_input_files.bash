#!/usr/bin/env bash
set -euo pipefail

# 1) Local, usando tu entorno Python del sistema
nextflow run main.nf -profile local \
  --script exp_ortel_parallel.py \
  --points "5,8,11,14,17,20" \
  --reps 2 \
  --N 3*10**5 --N_cp 500 --N_hip 5000

# 2) En SLURM con Singularity
nextflow run main.nf -profile slurm -resume \
  --container "python_39e84b5c10d96b50.sif" \
  --script exp_ortel_parallel.py \
  --points "5,8,11,14,17,20" \
  --reps 2 \
  --N 3*10**5 --N_cp 500 --N_hip 5000
