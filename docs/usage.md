# Usage

```bash
$ sourcepredict -h
usage: SourcePredict v0.33 [-h] [-a ALPHA] [-s SOURCES] [-l LABELS]
                           [-n NORMALIZATION] [-dt DISTANCE] [-me METHOD]
                           [-e EMBED] [-di DIM] [-o OUTPUT] [-se SEED]
                           [-k KFOLD] [-t THREADS]
                           otu_table

==========================================================
SourcePredict v0.33
Coprolite source classification
Author: Maxime Borry
Contact: <borry[at]shh.mpg.de>
Homepage & Documentation: github.com/maxibor/sourcepredict
==========================================================


positional arguments:
  otu_table         path to otu table in csv format

optional arguments:
  -h, --help        show this help message and exit
  -a ALPHA          Proportion of sink sample in unknown. Default = 0.1
  -s SOURCES        Path to source csv file. Default =
                    data/modern_gut_microbiomes_sources.csv
  -l LABELS         Path to labels csv file. Default =
                    data/modern_gut_microbiomes_labels.csv
  -n NORMALIZATION  Normalization method (RLE | Subsample | GMPR). Default =
                    GMPR
  -dt DISTANCE      Distance method. (unweighted_unifrac | weighted_unifrac)
                    Default = weighted_unifrac
  -me METHOD        Embedding Method. TSNE or UMAP. Default = TSNE
  -e EMBED          Output embedding csv file. Default = None
  -di DIM           Number of dimensions to retain for dimension reduction.
                    Default = 2
  -o OUTPUT         Output file basename. Default =
                    <sample_basename>.sourcepredict.csv
  -se SEED          Seed for random generator. Default = 42
  -k KFOLD          Number of fold for K-fold cross validation in parameter
                    optimization. Default = 5
  -t THREADS        Number of threads for parallel processing. Default = 2

```

## Running sourcepredict on the test dataset

```bash
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/test/dog_test_sample.csv -O dog_test_sample.csv
$ sourcepredict -t 6 dog_test_sample.csv
```

## Command line arguments

### -alpha

Proportion of alpha of sink sample in unknown. Default = `0.1`
$$\alpha \in [0,1]$$

*Example:*

`-alpha 0.1`

### -s SOURCES

Path to source `csv` (training) file with samples in columns, and OTUs in rows. Default = `data/sourcepredict/modern_gut_microbiomes_sources.csv`

*Example:*

`-s data/sourcepredict/modern_gut_microbiomes_sources.csv`

*Example source file format:*

```
+-------+----------+----------+
| TAXID | SAMPLE_1 | SAMPLE_2 |
+-------+----------+----------+
|  467  |    18    |    24    |
+-------+----------+----------+
|  786  |     3    |    90    |
+-------+----------+----------+
```

### -s LABELS

Path to labels `csv` file of sources.
Default = `data/modern_gut_microbiomes_labels.csv`

*Example:*

`-l data/modern_gut_microbiomes_labels.csv`

*Example source file format:*

```
+----------+--------+
|          | labels |
+----------+--------+
| SAMPLE_1 |   Dog  |
+----------+--------+
| SAMPLE_2 |  Human |
+----------+--------+
```

### -n NORMALIZATION  

Normalization method. One of `RLE`, `CLR`, `Subsample`, or `GMPR`. Default = `GMPR`

### -dt DISTANCE

Distance method. One of `unweighted_unifrac`, `weighted_unifrac`. Default = `weighted_unifrac`

_Example:_

`-dt weighted_unifrac`

### -me METHOD

Embedding Method. One of `TSNE` or `UMAP`. Default = `TSNE`

_Example:_

`-me TSNE`

### -e EMBED

File for saving embedding coordinates in `csv` format. Default = `None`

_Example:_

`-e embed_coord.csv`

### -di DIM

Number of dimensions to retain for dimension reduction by embedding with UMAP or TSNE. Default = `2`

_Example:_

`-di 2`

### -o OUTPUT

Sourcepredict Output file basename. Default = `<sample_basename>.sourcepredict.csv`

_Example:_

`-o my_output`

### -se SEED

Seed for random number generator. Default = `42`

_Example:_

`-se 42`

### -k KFOLD

Number of fold for K-fold cross validation in in parameter optimization. Default = `5`

_Example:_

`-k 5`

### -t THREADS

Number of threads for parallel processing. Default = `2`

_Example:_

`-t 2`
