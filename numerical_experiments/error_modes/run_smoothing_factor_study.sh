#!/bin/bash

gamma=.1
iterations=10

for ((i=0; i<iterations;i++))
do
    python3 smoothing_factor_all.py $gamma

    dir_name=gamma_"$gamma"
    mkdir $dir_name
    mv *pdf $dir_name

    gamma=$(echo "$gamma+0.1"|bc)

done
