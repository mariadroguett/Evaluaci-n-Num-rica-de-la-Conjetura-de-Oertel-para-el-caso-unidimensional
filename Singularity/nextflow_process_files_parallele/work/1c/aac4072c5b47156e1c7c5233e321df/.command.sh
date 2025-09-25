#!/bin/bash -ue
set -euo pipefail
mkdir -p results_nextflow
ts=$(date +%Y%m%d_%H%M%S)
out="exp_stub_20pts_rep1_${ts}.csv"
echo 'seed,F,bestCP' > "${out}"
echo '0,0.0,[]' >> "${out}"
