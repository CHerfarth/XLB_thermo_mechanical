#!/bin/bash

nodes_x=16
nodes_y=16
timesteps_mg=100
timesteps_standard=10000
coarsest_level_iter=5000
iterations=4

E=0.4
nu=0.5

for ((i=0; i<iterations; i++))
do
    python3 convergence_tester.py $nodes_x $nodes_y $timesteps_mg $timesteps_standard $coarsest_level_iter $E $nu | tee tmp_1.txt
    cat tmp_1.txt > log.txt


    cat tmp_1.txt | grep "E_scaled" > tmp_2.txt
    E_scaled=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    cat tmp_1.txt | grep "nu" > tmp_2.txt
    nu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    #get amplification factor
    python3 ../error_modes/smoothing_factor_single.py $E_scaled $nu > tmp_1.txt
    cat tmp_1.txt > log_2.txt

    cat tmp_1.txt | grep "Amplification" > tmp_2.txt
    amplification=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    cat tmp_1.txt | grep "Smoothing" > tmp_2.txt
    smoothing=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    echo "Expected amplification factor $amplification"
    #echo "Expected smoothing factor standard LB $smoothing"

    python3 plotter.py $amplification 2 $smoothing


    rm tmp*

    dir=nodes_$nodes_x
    mkdir $dir
    mv *.png $dir

    echo "============ Iteration $i done =============="
    echo ""

    nodes_x=$((nodes_x*2))
    nodes_y=$((nodes_y*2))
    coarsest_level_iter=$((coarsest_level_iter*4))

done



