#!/usr/bin/env bash
set -euo pipefail

# 1) Local, usando tu entorno Python del sistema
nextflow run main.nf -profile local \
  --script exp_ortel_parallel.py \
  --points "5,8,11,14,17,20" \
  --reps 2 \
  --N 100000 --N_cp 50 --N_hip 500

# 2) En SLURM con Singularity
# Ojo: ajusta el nombre del .sif que tienes en la carpeta
nextflow run main.nf -profile slurm -resume \
  --container "python_39e84b5c10d96b50.sif" \
  --script exp_ortel_parallel.py \
  --points "5,8,11,14,17,20" \
  --reps 2 \
  --N 100000 --N_cp 50 --N_hip 500
