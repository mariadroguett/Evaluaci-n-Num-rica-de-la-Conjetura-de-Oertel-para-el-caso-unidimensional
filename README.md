# Evaluación Numérica de la Conjetura de Oertel para el caso unidimensional

## SLURM

Se incluyen scripts en `slurm/` para ejecutar en un cluster con SLURM:

- `slurm/ortel_single.sbatch`: una sola corrida (`python main.py`).
- `slurm/ortel_parallel.sbatch`: múltiples semillas en paralelo usando `exp_ortel_parallel.py` (pool de procesos interno).
- `slurm/ortel_array.sbatch`: job array; cada tarea procesa una semilla distinta.

Ajusta flags SBATCH (tiempo, memoria, partición, cuenta) y, si hace falta, carga de módulos/entorno de Python.

Ejemplos:

- Paralelo interno, 8 CPUs por tarea:
  `sbatch slurm/ortel_parallel.sbatch`
  (puedes sobreescribir variables: `SEEDS`, `OUTDIR`, `FIBERS_DIR`, `CP_OUT`, `WORKERS`)

- Job array de 10 tareas (editando `--array`), con semillas `BASE_SEED+ID`:
  `sbatch slurm/ortel_array.sbatch`

Notas:
- Los scripts exportan `OMP_NUM_THREADS=1` y similares para evitar sobre-suscripción de BLAS/OpenMP.
- `exp_ortel_parallel.py` ya admite `--fibers-dir` (guardar/cargar fibras por semilla), `--reuse-fibers`, y `--cp-out` (guardar CPs aceptados por semilla).
