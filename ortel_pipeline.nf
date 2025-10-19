#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// ==================== PARÁMETROS ====================
if( !params.containsKey('num_polytopes') )   params.num_polytopes   = 0
if( !params.containsKey('seeds') )          params.seeds           = []
if( !params.containsKey('n_per_z') )        params.n_per_z         = 5
if( !params.containsKey('d') )              params.d               = 2
if( !params.containsKey('z_vals') )         params.z_vals          = [0, 1]
if( !params.containsKey('N') )              params.N               = 100000
if( !params.containsKey('N_cp') )           params.N_cp            = 100
if( !params.containsKey('N_hip') )          params.N_hip           = 3000
if( !params.containsKey('batch') )          params.batch           = 5000
if( !params.containsKey('triangle') )       params.triangle        = true
if( !params.containsKey('experiments_csv')) params.experiments_csv = "results/experiments.csv"
if( !params.containsKey('save_fibers_dir')) params.save_fibers_dir = "results/fibras"
if( !params.containsKey('f_threshold') )    params.f_threshold     = 0.18
if( !params.containsKey('n_point_values'))  params.n_point_values  = [5, 8]
if( !params.containsKey('save_hull_dir') )  params.save_hull_dir   = "results/hulls"
if( !params.containsKey('container') )      params.container       = null

def seeds_list
if( params.seeds instanceof String ) {
    seeds_list = params.seeds.tokenize(', ').collect { it.isInteger() ? it.toInteger() : it }
} else if( params.seeds instanceof Collection ) {
    seeds_list = params.seeds as List
} else if( params.seeds ) {
    seeds_list = [params.seeds]
} else {
    seeds_list = []
}

def num_polys = params.num_polytopes instanceof String && params.num_polytopes.isInteger()
        ? params.num_polytopes.toInteger()
        : (params.num_polytopes ?: 0) as int

if( seeds_list.isEmpty() ) {
    def total = num_polys > 0 ? num_polys : 1
    seeds_list = (1..total).collect { it }
}

def z_vals_list
if( params.z_vals instanceof String ) {
    z_vals_list = params.z_vals.tokenize(', ').findAll { it }.collect {
        try { it.toInteger() } catch (e) {
            try { it.toDouble() } catch (e2) { it }
        }
    }
} else {
    z_vals_list = params.z_vals instanceof Collection ? params.z_vals as List : [params.z_vals]
}
params.z_vals = z_vals_list

def npoint_list
if( params.n_point_values instanceof String ) {
    npoint_list = params.n_point_values.tokenize(', ').findAll { it }.collect {
        it.isInteger() ? it.toInteger() : it.toDouble()
    }
} else if( params.n_point_values instanceof Collection ) {
    npoint_list = params.n_point_values as List
} else {
    npoint_list = [params.n_point_values]
}
params.n_point_values = npoint_list

params.seeds = seeds_list
params.num_polytopes = num_polys

// ==================== LOG INICIAL ====================
println "Iniciando pipeline ORTEL con semillas: ${params.seeds}"
println "Parámetros: n_per_z=${params.n_per_z}, d=${params.d}, N=${params.N}, N_cp=${params.N_cp}, N_hip=${params.N_hip}"
if (params.triangle) {
    println "Modo triángulo activado: usando triángulo estándar en R²."
    if (params.d && params.d != 2) {
        println "Nota: se fuerza d=2 para modo triángulo."
    }
} else {
    println "Modo politopo aleatorio: z_vals=${params.z_vals}, n_per_z=${params.n_per_z}, d=${params.d}"
}

// ==================== CANALES ====================
n_points_ch = Channel.of(params.n_point_values)
seeds_ch = Channel.of(params.seeds)
combo_ch = seeds_ch.cross(n_points_ch)
def total_runs = params.seeds.size() * params.n_point_values.size()

// ==================== PROCESO ORTEL ====================
process run_ortel {
    tag "seed_${seed}_npoint_${n_point}"
    echo true
    if( params.container ) container params.container

    input:
        tuple val(seed), val(n_point) from combo_ch
        file 'main_ortel.py'     from file("${projectDir}/main_ortel.py")
        file 'convex_hull.py'   from file("${params.repo_root ?: projectDir}/convex_hull.py")
        file 'ortel.py'         from file("${params.repo_root ?: projectDir}/ortel.py")
        file 'vol_reject.py'    from file("${params.repo_root ?: projectDir}/vol_reject.py")
        file 'vol_star.py'      from file("${params.repo_root ?: projectDir}/vol_star.py")

    output:
        file "seed_${seed}_npoint_${n_point}.csv"

    script:
        def seed_idx = params.seeds.indexOf(seed)
        def npoint_idx = params.n_point_values.indexOf(n_point)
        def index = seed_idx * params.n_point_values.size() + npoint_idx + 1
        """
        echo "[${index}/${total_runs}] Ejecutando experimento con semilla=${seed} y n_point=${n_point}"
        export PYTHONPATH=\$(pwd):\$PYTHONPATH

        python main_ortel.py --seed ${seed} \\
            ${params.n_per_z     ? "--n_per_z ${params.n_per_z}" : ""} \\
            ${params.d           ? "--d ${params.d}" : ""} \\
            ${!params.triangle && params.z_vals ? "--z_vals \"${params.z_vals.join(',')}\"" : ""} \\
            ${params.N           ? "--N ${params.N}" : ""} \\
            ${params.N_cp        ? "--N_cp ${params.N_cp}" : ""} \\
            ${params.N_hip       ? "--N_hip ${params.N_hip}" : ""} \\
            ${params.batch       ? "--batch ${params.batch}" : ""} \\
            ${params.triangle    ? "--triangle" : ""} \\
            --save_fibers_dir ${params.save_fibers_dir} \\
            --f_threshold ${params.f_threshold} \\
            --n_point ${n_point} \\
            --save_hull_dir ${params.save_hull_dir} \\
            --out "seed_${seed}_npoint_${n_point}.csv"
        """
}

// ==================== MERGE RESULTADOS ====================
process merge_results {
    publishDir "./", mode: 'overwrite'
    echo true

    input:
        file seed_csvs from run_ortel.out.collect()

    output:
        file("${params.experiments_csv}")

    script:
    """
    echo "Combinando resultados en ${params.experiments_csv}"
    out_file=${params.experiments_csv}
    header_saved=false
    > \$out_file
    for f in ${seed_csvs}; do
        if [ "\$header_saved" = false ]; then
            cat "\$f" >> \$out_file
            header_saved=true
        else
            tail -n +2 "\$f" >> \$out_file
        fi
    done
    echo "Combinados \$(ls ${seed_csvs} | wc -l) archivos."
    """
}
