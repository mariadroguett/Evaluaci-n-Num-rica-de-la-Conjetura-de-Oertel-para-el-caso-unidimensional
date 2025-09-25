#!/bin/bash -ue
set -euo pipefail
mkdir -p results_nextflow
ts=$(date +%Y%m%d_%H%M%S)
out="results_nextflow/exp_stub_5pts_rep1_${ts}.csv"
echo 'seed,F,bestCP' > "${out}"
echo '0,0.0,[]' >> "${out}"
