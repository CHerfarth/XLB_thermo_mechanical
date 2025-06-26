#!/bin/bash

timesteps_mg=100
coarsest_level_iter=500
iterations=6

v1=2
v2=1
v_tot=$((v1+v2))

base_E=.1
base_nu=.2
vals_E=5
vals_nu=4

gamma=0.8


for ((j=0; j<vals_E; j++))
do
    for ((k=0; k<vals_nu; k++))
    do
        E=$(echo "$base_E + $j*0.2"|bc -l)
        nu=$(echo "$base_nu + 0.2*$k"|bc -l)
        nodes_x=16
        nodes_y=16
        for ((i=0; i<iterations; i++))
        do
            python3 convergence_tester.py $nodes_x $nodes_y $timesteps_mg $coarsest_level_iter $E $nu $v1 $v2 | tee tmp_1.txt
            cat tmp_1.txt > log.txt


            cat tmp_1.txt | grep "E_scaled" > tmp_2.txt
            E_scaled=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

            cat tmp_1.txt | grep "nu" > tmp_2.txt
            nu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

            #get amplification factor
            python3 ../error_modes/smoothing_factor_single.py $E_scaled $nu $gamma > tmp_1.txt
            cat tmp_1.txt > log_2.txt

            cat tmp_1.txt | grep "Smoothing" > tmp_2.txt
            smoothing=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

            echo "Expected smoothing factor $smoothing"

            python3 plotter.py $v_tot $smoothing $E $nu


            rm tmp*

            dir=nodes_$nodes_x
            mkdir $dir
            mv *.pdf $dir

            echo "============ Iteration $i done =============="
            echo ""

            nodes_x=$((nodes_x*2))
            nodes_y=$((nodes_y*2))
            #coarsest_level_iter=$((coarsest_level_iter*4))

        done
        dir=E_"$E"_nu_"$nu"
        mkdir $dir
        mv nodes_* $dir
    done
done



