#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process processFiles {
    // Define the input files for this process
    
    publishDir "outDir_nextflow/" , mode : "copy"
    input:
    path files

    // Define the output
    output:
    path "processed_${files}"


    // Define the script to run
    script:
    """
    echo "Processing files: ${files}"
    # Here you can add your processing commands
    # For demonstration, we will just create a processed file(change the script file given your own location)
    python /mnt/beegfs/home/dgonzalez/singularity_slurm_nextflow/nextflow_process_files_parallele/process_file.py  ${files} processed_${files}
    """
}

// Define the main workflow
workflow {
    // Get all text files
    files = Channel.fromPath('input_files/file_*.txt')
    
    //Divide the 100 files in groups of 25 files
    grouped_files = files.buffer(size: 25)

    //Run in parallele
    grouped_files.flatMap { batch ->
        batch
    } | processFiles
}
