#!/usr/bin/env nextflow

params.reads = ''
params.krakendb = '/path/to/minikraken_20171101_4GB_dustmasked/'
params.phred = 33
params.results = './results'
params.pairedEnd = true

Channel
    .fromFilePairs( params.reads, size: params.pairedEnd ? 2 : 1 )
    .ifEmpty { exit 1, "Cannot find any reads matching: ${params.reads}\n" }
	.set {reads_to_trim}


process AdapterRemoval {
    tag "$name"

    conda 'bioconda::adapterremoval'

    label 'expresso'

    input:
        set val(name), file(reads) from reads_to_trim

    output:
        set val(name), file('*.trimmed.fastq') into trimmed_reads
        file("*.settings") into adapter_removal_results

    script:
        out1 = name+".pair1.trimmed.fastq"
        out2 = name+".pair2.trimmed.fastq"
        se_out = name+".trimmed.fastq"
        settings = name+".settings"
        if (params.pairedEnd){
            """
            AdapterRemoval --basename $name --file1 ${reads[0]} --file2 ${reads[1]} --trimns --trimqualities --minquality 20 --minlength 30 --output1 $out1 --output2 $out2 --threads ${task.cpus} --qualitybase ${params.phred} --settings $settings
            """
        } else {
            """
            AdapterRemoval --basename $name --file1 ${reads[0]} --trimns --trimqualities --minquality 20 --minlength 30 --output1 $se_out --threads ${task.cpus} --qualitybase ${params.phred} --settings $settings
            """
        }
            
}

process miniKraken {
    tag "$name"

    conda 'bioconda::kraken'

    label 'intenso'

    input:
        set val(name), file(reads) from trimmed_reads

    output:
        set val(name), file('*.kraken.out') into kraken_out

    script:
        out = name+".kraken.out"
        if (params.pairedEnd){
            """
            kraken --db ${params.krakendb} --threads ${task.cpus} --preload --fastq-input --paired --output $out  ${reads[0]} ${reads[1]}
            """    
        } else {
            """
            kraken --db ${params.krakendb} --threads ${task.cpus} --preload --fastq-input --output $out  ${reads[0]}
            """
        }
        
}

process Kraken_report {
    tag "$name"

    conda 'bioconda::kraken'

    label 'ristretto'

    publishDir "${params.results}/kraken", mode: 'copy'

    input:
        set val(name), file(kraken_o) from kraken_out

    output:
        set val(name), file('*.kraken_report.out') into kraken_report

    script:
        report = name+".kraken_report.out"
        """
        kraken-report --db ${params.krakendb} $kraken_o > $report
        """    
}

process kraken_parse {
    tag "$name"

    conda 'python=3.6'

    label 'ristretto'

    input:
        set val(name), file(kraken_r) from kraken_report

    output:
        set val(name), file('*.kraken_parsed.csv') into kraken_parsed

    script:
        out = name+".kraken_parsed.csv"
        """
        kraken_parse.py $kraken_r
        """    
}

process kraken_merge {

    conda 'python=3.6 pandas numpy'

    label 'ristretto'

    publishDir "${params.results}/merged", mode: 'copy'

    input:
        file(csv_count) from kraken_parsed.collect()

    output:
        file('kraken_merged_*.csv') into kraken_merged

    script:
        out = "kraken_merged.csv"
        """
        merge_kraken_res.py -o $out
        """    
}