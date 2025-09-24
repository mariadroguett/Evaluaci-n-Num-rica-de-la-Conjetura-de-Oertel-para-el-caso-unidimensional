#!/bin/bash
#SBATCH --job-name=optim_sing           # Nombre del job (aparece en squeue)
#SBATCH --output=logs_slurm/optim_sing_%j.out   # Log de salida (%j = jobID)
#SBATCH --error=logs_slurm/optim_sing_%j.err    # Log de errores
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Tareas (1 = un solo proceso principal)
#SBATCH --cpus-per-task=20              # Núcleos de CPU por tarea
#SBATCH --time=06:00:00                 # Tiempo máximo (hh:mm:ss)
#SBATCH --mem=32G                       # Memoria total (ajusta si necesitas más)
#SBATCH --partition=ngen-ko             # Cola/partición (cámbiala si usas otra)

# --- Carga de módulos opcional ---
# module load singularity

# --- Variables a ajustar ---
WORKDIR=/home/$USER/Evaluaci-n-Num-rica-de-la-Conjetura-de-Oertel-para-el-caso-unidimensional/Singularity/nextflow
IMG=$WORKDIR/python_39e84b5c10d96b50.sif      # Tu imagen .sif
CMD="nextflow run main.nf -profile slurm -resume \
      --container $IMG \
      --script exp_ortel_parallel.py \
      --points '5,8,11,14,17,20' \
      --reps 2 \
      --N 100000 --N_cp 50 --N_hip 500"

# --- Ejecutar ---
cd "$WORKDIR"
singularity exec -B "$WORKDIR" "$IMG" $CMD
