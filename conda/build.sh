#!/bin/bash

cp sourcepredict $PREFIX/bin
cp -r sourcepredictlib $PREFIX/bin
mkdir -p $PREFIX/bin/data
cp data/*.csv $PREFIX/bin/data


