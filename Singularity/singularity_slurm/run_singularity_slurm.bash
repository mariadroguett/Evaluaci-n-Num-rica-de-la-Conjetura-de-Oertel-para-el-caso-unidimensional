#!/bin/bash
# Fail fast
set -euo pipefail
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
# Directorio con los archivos de Nextflow de este repo
WORKDIR=/home/$USER/Evaluaci-n-Num-rica-de-la-Conjetura-de-Oertel-para-el-caso-unidimensional/Singularity/nextflow_process_files_parallele
IMG=$WORKDIR/python_39e84b5c10d96b50.sif      # Tu imagen .sif (ajusta el nombre si difiere)

# --- Ejecutar ---
cd "$WORKDIR"
# Asegura que exista el directorio de logs (recomendado crearlo antes de sbatch también)
mkdir -p logs_slurm

# Asegura que Python vea el repo (para imports como convex_hull, ortel)
export PYTHONPATH="${WORKDIR}/../..:${PYTHONPATH:-}"

# Construye el comando Nextflow; agrega contenedor solo si existe la imagen
CMD=( nextflow run main.nf -profile slurm -resume \
  --script exp_ortel_parallel.py \
  --points '5,8,11,14,17,20' \
  --reps 2 \
  --N 100000 --N_cp 50 --N_hip 500 )

if [[ -f "$IMG" ]]; then
  CMD+=( --container "$IMG" )
else
  echo "[WARN] Imagen SIF no encontrada en $IMG. Ejecutando sin contenedor." >&2
fi

"${CMD[@]}"
