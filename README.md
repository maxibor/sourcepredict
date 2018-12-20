[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict)

# SourcePredict

Prediction/source tracking of sample source using a random forest approach
## Help

```
usage: SourcePredict v0.1 [-h] [-a ALPHA] [-s SOURCES] [-n NORMALIZATION]
                          [-o OUTPUT] [-se SEED] [-t THREADS]
                          otu_table

==========================================================
SourcePredict v0.1
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
                    ./data/dog_human_pig_sources.csv
  -n NORMALIZATION  Normalization method (RLE | CLR | Subsample). Default =
                    Subsample
  -o OUTPUT         Output file basename. Default =
                    <sample_basename>.sourcepredict.csv*
  -se SEED          Seed for random generator. Default = None (randomly
                    generated)
  -t THREADS        Number of threads for parallel processing. Default = 2
```
