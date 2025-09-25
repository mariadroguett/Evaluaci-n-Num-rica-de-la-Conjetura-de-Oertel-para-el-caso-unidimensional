nextflow.enable.dsl=2

/***********************
 * Parámetros por defecto
 ***********************/
params.d            = 2
params.z_vals       = '0,1'
params.N            = 100_000
params.N_cp         = 50
params.N_hip        = 500
params.batch        = 5000
params.guided       = false
params.do_merge     = false

params.points       = '5,8,11,14,17,20'   // N° de puntos por fibra
params.reps         = 2                   // repetir cada caso
params.seeds        = ''                  // opcional: "101,102,103"; si vacío, se autogenera

params.script       = 'exp_ortel_parallel.py' // tu script Python
params.outdir       = 'results_nextflow'      // salida
params.container    = ''                      // p.ej. "python_39e84b5c10d96b50.sif"

// Estos tres existían en tu archivo y daban WARN si faltaban:
params.fibers_dir   = 'fibers'
params.reuse_fibers = false
params.cp_out       = 'cp_results'

/***********************
 * Helpers
 ***********************/
def parseIntList(def s){ s.toString().split(',').findAll{ it }.collect{ it as int } }
def parseStrList(def s){ s == null ? [] : s.toString().split(',').findAll{ it }.collect{ it.trim() } }

/***********************
 * Generación de combinaciones (npts × rep × seed)
 ***********************/
Channel
  .from( parseIntList(params.points) )
  .flatMap { npts -> (1..params.reps).collect { rep -> tuple(npts as int, rep as int) } }
  .map { pair ->
      def npts = pair[0] as int
      def rep  = pair[1] as int
      def fixedSeeds = parseStrList(params.seeds)
      if( fixedSeeds ) {
        def s = (fixedSeeds[(rep-1) % fixedSeeds.size()] as int)
        tuple(npts, rep, s)
      }
      else {
        def base = (System.currentTimeMillis() + (long)(npts*100 + rep)) as long
        def s = (base & 0x7fffffffL) as int
        tuple(npts, rep, s)
      }
  }
  .set { COMBOS }                          // (npts, rep, seed)

/***********************
 * Proceso principal: corre tu script Python
 ***********************/
process RUN_EXPERIMENT {
  tag "${npts}pts_rep${rep}"
  publishDir params.outdir, mode: 'copy', overwrite: false

  // El contenedor se gestiona vía nextflow.config (perfil slurm)

  input:
  tuple val(npts), val(rep), val(seed)

  output:
  path "exp_*_${npts}pts_rep${rep}_*.json", emit: jsons

  script:
  """
  set -euo pipefail

  mkdir -p ${params.outdir} tmp_logs
  export PYTHONPATH="${projectDir}/../..:\${PYTHONPATH:-}"

  JSON_OUT="exp_${seed}_${npts}pts_rep${rep}_\$(date +%Y%m%d_%H%M%S).json"
  python ${projectDir}/${params.script} \
    --d ${params.d} \
    --z_vals ${params.z_vals} \
    --n_per_z ${npts} \
    --N ${params.N} \
    --N_cp ${params.N_cp} \
    --N_hip ${params.N_hip} \
    --batch ${params.batch} \
    ${params.guided ? '--guided' : ''} \
    --seed ${seed} \
    > "\$JSON_OUT" 2> "tmp_logs/run_${npts}pts_rep${rep}.log"
  """

  stub:
  """
  set -euo pipefail
  ts=\$(date +%Y%m%d_%H%M%S)
  out="exp_stub_${npts}pts_rep${rep}_\${ts}.json"
  echo '{"seed":0,"F":0.0,"bestCP":[]}' > "\${out}"
  """
}

/**
 * Merge opcional de JSONs a JSONL
 */
process MERGE_JSON {
  tag "merge_json"
  publishDir params.outdir, mode: 'copy', overwrite: true

  input:
  path json_files

  output:
  path "all_results_\\d{8}_\\d{6}.jsonl"

  shell:
  """
  set -euo pipefail
  ts=\$(date +%Y%m%d_%H%M%S)
  cat !{json_files} > "all_results_\${ts}.jsonl"
  """
}

/***********************
 * Workflow
 ***********************/
workflow {
  main:
    results = RUN_EXPERIMENT( COMBOS )
    if( params.do_merge ) {
      MERGE_JSON( results.jsons.collect() )
    }

  emit:
    jsons  = results.jsons
    merged = params.do_merge ? MERGE_JSON.out : Channel.empty()
}
