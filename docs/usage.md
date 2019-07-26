# Usage



## Running sourcepredict on the test dataset

```
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/test/dog_test_sample.csv -O dog_example.csv
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/modern_gut_microbiomes_labels.csv -O sp_labels.csv
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/modern_gut_microbiomes_sources.csv -O sp_sources.csv
$ sourcepredict -s sp_sources.csv -l sp_labels.csv dog_example.csv
```

## Command line interface

```bash
$ sourcepredict -h
usage: SourcePredict v0.33 [-h] [-a ALPHA] [-s SOURCES] [-l LABELS]
                           [-n NORMALIZATION] [-dt DISTANCE] [-me METHOD]
                           [-kne NEIGHBORS] [-kw WEIGHTS] [-e EMBED] [-di DIM]
                           [-o OUTPUT] [-se SEED] [-k KFOLD] [-t THREADS]
                           abundance_table

==========================================================
SourcePredict v0.33
Coprolite source classification
Author: Maxime Borry
Contact: <borry[at]shh.mpg.de>
Homepage & Documentation: github.com/maxibor/sourcepredict
==========================================================


positional arguments:
  abundance_table   path to TAXID abundance count table in csv format

optional arguments:
  -h, --help        show this help message and exit
  -a ALPHA          Proportion of sink sample in unknown. Default = 0.1
  -s SOURCES        Path to source csv file. Default =
                    data/modern_gut_microbiomes_sources.csv
  -l LABELS         Path to labels csv file. Default =
                    data/modern_gut_microbiomes_labels.csv
  -n NORMALIZATION  Normalization method (RLE | Subsample | GMPR | None).
                    Default = GMPR
  -dt DISTANCE      Distance method. (unweighted_unifrac | weighted_unifrac)
                    Default = weighted_unifrac
  -me METHOD        Embedding Method. TSNE, MDS, or UMAP. Default = TSNE
  -kne NEIGHBORS    Numbers of neigbors for KNN classication. Default = 0
                    (chosen by CV)
  -kw WEIGHTS       Sample weight function for KNN prediction (distance |
                    uniform). Default = distance.
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

## Command line arguments

### abundance_table

Sink abundance_table in `csv` file format

*Example sink count table file*

```
+-------+----------+----------+
| TAXID |  SINK_1  |  SINK_2  |
+-------+----------+----------+
|  283  |    5     |    2     |
+-------+----------+----------+
|  143  |     25   |    48    |
+-------+----------+----------+
```

### -alpha

Proportion of alpha of sink sample in unknown. Default = `0.1`
$$\alpha \in [0,1]$$

*Example:*

`-alpha 0.1`

### -s SOURCES

Path to source `csv` (training) file with samples in columns, and TAXIDs in rows. Default = `data/sourcepredict/modern_gut_microbiomes_sources.csv`

*Example:*

`-s data/sourcepredict/modern_gut_microbiomes_sources.csv`

*Example source file :*

```
+-------+----------+----------+
| TAXID | SAMPLE_1 | SAMPLE_2 |
+-------+----------+----------+
|  467  |    18    |    24    |
+-------+----------+----------+
|  786  |     3    |    90    |
+-------+----------+----------+
```

### -l LABELS

Path to labels `csv` file of sources.
Default = `data/modern_gut_microbiomes_labels.csv`

*Example:*

`-l data/modern_gut_microbiomes_labels.csv`

*Example source file :*

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

### -kne NEIGHBORS

Numbers of neigbors for KNN classication. Default = 0 (chosen by CV).

_Example:_

`-kne 30`

> Setting the number of neighbors to 0 will let Sourcepredict choose the optimal number of neighbors for **classification**.  
However, for **source proportion estimation**, setting manually a higher number of samples (for example, 50) will help for better proportion estimations.  
See [example 2](https://sourcepredict.readthedocs.io/en/latest/mixed_prop.html) for illustration.

### --kw WEIGHTS

Sample weight function for KNN prediction (distance | uniform). Default = distance.

Choose to give a uniform or distance based weights to neighbor samples in KNN algorithm.

> Distance base weights will work better for **classification** while uniform weigths will work better for **source proportion estimation**.  
See [example 2](https://sourcepredict.readthedocs.io/en/latest/mixed_prop.html) for illustration.

### -e EMBED

File for saving embedding coordinates in `csv` format. Default = `None`

_Example:_

`-e embed_coord.csv`

### -di DIM

Number of dimensions to retain for dimension reduction. Default = `2`

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

Number of fold for K-fold cross validation in parameter optimization. Default = `5`

_Example:_

`-k 5`

### -t THREADS

Number of threads for parallel processing. Default = `2`

_Example:_

`-t 2`



## Choice of the taxonomic classifier

Different taxonomic classifiers will give different results, because of different algorithms, and different databases.

In order to produce correct results with Sourcepredict, **the taxonomic classifier used to produce the *source* TAXID count table must be the same as the one used to produce the *sink* TAXID count table**.

Because Sourcepredict relies on machine learning, at least 10 samples per sources are required, but more source samples will lead to a better prediction by Sourcepredict.

Therefore, running all these samples through a taxonomic classifier ahead of Sourcepredict requires a non-negligeable computational time.

Hence the choice of the taxonomic classifier is a balance between precision, and computational time. 

While this documentation doesn't intent to be a benchmark of taxonomic classifiers, the author of Sourcepredict has had decent results with [Kraken2](https://ccb.jhu.edu/software/kraken2/) and recommends it for its good compromise between precision and runtime.

The example *source* and *sink* data provided with Sourcepredict were generated with Kraken2.






