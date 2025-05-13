#!/bin/bash

gamma=0.8
iterations=1

for ((i=0; i<iterations;i++))
do
    python3 smoothing_factor_all.py $gamma > tmp.txt

    dir_name=gamma_"$gamma"
    mkdir $dir_name
    mv *png $dir_name

    gamma=$(echo "$gamma+0.1"|bc)
    rm tmp.txt

done
