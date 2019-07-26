# Custom sources

> Different taxonomic classifers will give different results, and **the taxonomic classifier used to produce the *source* OTU count table must be the same as the one used to produce the *sink* OTU count table**.

While there are many available taxonomic classifiers available to produce the source and sink OTU table, the Sourcepredict author provide a simple pipeline to generate the source and sink OTU table.

This pipeline is written using [Nextflow](https://www.nextflow.io/), and handles the dependancies using [conda](https://conda.io/en/latest/).
Briefly, this pipelines will firt trim and clip the sequencing files with [AdapterRemoval](https://github.com/MikkelSchubert/adapterremoval) before performing the taxonomic classification with [Kraken2](https://ccb.jhu.edu/software/kraken2).

## Pipeline installation

```
$ conda install -c bioconda nextflow
$ nextflow pull maxibor/kraken-nf
```

## Running the pipeline

See the [README](https://github.com/maxibor/kraken-nf) of [maxibor/kraken-nf](https://github.com/maxibor/kraken-nf)