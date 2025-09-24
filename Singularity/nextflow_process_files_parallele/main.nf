nextflow.enable.dsl=2

params.seeds     = (params.seeds ?: '105,106,107,108,109')    // --seeds "..."
params.workers   = (params.workers ?: 5)                      // --workers 5
params.outdir    = (params.outdir ?: 'results')               // --outdir ...
params.fibersdir = (params.fibers_dir ?: null)                // --fibers-dir ...
params.reuse     = (params.reuse_fibers ?: false)             // --reuse-fibers
params.cpout     = (params.cp_out ?: null)                    // --cp-out ...
params.container = (params.container ?: null)                 // --container image.sif
params.script    = (params.script ?: 'exp_main.py')           // tu script Python

// Fan-out por seed
Channel
  .of( params.seeds.split(',').collect{ it.trim() }.findAll{ it } )
  .map { it as Integer }
  .set { SEEDS }

process RUN_EXP {
  tag "seed:${seed}"
  publishDir params.outdir, mode: 'copy', overwrite: true

  cpus  params.workers
  memory '16 GB'
  time  '12h'

  // Usa Singularity si pasas --container
  container params.container ?: null

  input:
  val seed from SEEDS

  output:
  path "seed_${seed}.json"

  script:
  def fibArg = params.fibersdir ? "--fibers-dir ${params.fibersdir}" : ""
  def cpArg  = params.cpout ? "--cp-out ${params.cpout}" : ""
  def reuse  = params.reuse ? "--reuse-fibers" : ""
  """
  set -euo pipefail
  mkdir -p ${params.outdir}
  python ${params.script} \
    --seeds "${seed}" \
    --workers ${params.workers} \
    --outdir ${params.outdir} \
    ${fibArg} ${reuse} ${cpArg}
  """
}

workflow {
  main:
    RUN_EXP(SEEDS)
}
