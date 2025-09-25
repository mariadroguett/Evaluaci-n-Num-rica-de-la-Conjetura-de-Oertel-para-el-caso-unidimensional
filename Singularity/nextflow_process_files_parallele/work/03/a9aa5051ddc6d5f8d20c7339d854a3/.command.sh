#!/bin/bash -ue
set -euo pipefail

mkdir -p results_nextflow tmp_logs

python exp_ortel_parallel.py     --d 2     --z_vals 0,1     --n_per_z 11     --N 100000     --N_cp 50     --N_hip 500     --batch 5000          --seed -2130917754     --out "exp_-2130917754_11pts_rep2_$(date +%Y%m%d_%H%M%S).csv"     > "tmp_logs/run_11pts_rep2.log" 2>&1
