#!/bin/bash -ue
set -euo pipefail

mkdir -p results_nextflow tmp_logs
export PYTHONPATH="/home/maria/Evaluaci-n-Num-rica-de-la-Conjetura-de-Oertel-para-el-caso-unidimensional/Singularity/nextflow_process_files_parallele/../..:${PYTHONPATH:-}"

JSON_OUT="exp_-2127574493_8pts_rep1_$(date +%Y%m%d_%H%M%S).json"
python exp_ortel_parallel.py     --d 2     --z_vals 0,1     --n_per_z 8     --N 100000     --N_cp 50     --N_hip 500     --batch 5000          --seed -2127574493     > "$JSON_OUT" 2> "tmp_logs/run_8pts_rep1.log"
