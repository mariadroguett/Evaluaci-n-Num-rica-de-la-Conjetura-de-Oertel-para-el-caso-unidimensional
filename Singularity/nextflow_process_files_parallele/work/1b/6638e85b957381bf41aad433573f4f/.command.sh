#!/bin/bash -ue
set -euo pipefail

mkdir -p results_nextflow tmp_logs

python exp_ortel_parallel.py     --d 2     --z_vals 0,1     --n_per_z 17     --N 100000     --N_cp 50     --N_hip 500     --batch 5000          --seed -2131200853     --out "exp_${NF_TASK_ID}_17pts_rep1_$(date +%Y%m%d_%H%M%S).csv"     > "tmp_logs/run_17pts_rep1.log" 2>&1
