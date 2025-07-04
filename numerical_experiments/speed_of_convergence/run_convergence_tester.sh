#!/bin/bash

timesteps_mg=100
coarsest_level_iter=500
iterations=2


base_E=.1
base_nu=.2
vals_E=5
vals_nu=4

gamma=0.8


v1=1
v2=1
vals_v=5

results_file="speed_of_convergence_results.csv"
echo "nodes_x,nodes_y,E,nu,v1,v2,smoothing_pow,convergence_factor,actual_convergence" > $results_file

for ((l=0; l<vals_v; l++))
do
    v_tot=$((v1+v2))
    for ((j=0; j<vals_E; j++))
    do
        for ((k=0; k<vals_nu; k++))
        do
            E=$(echo "$base_E + $j*0.2"|bc -l)
            nu=$(echo "$base_nu + 0.2*$k"|bc -l)

            #get smoothing factor
            python3 ../error_modes/smoothing_factor_single.py $E $nu $gamma > tmp_1.txt
            cat tmp_1.txt > log_2.txt

            cat tmp_1.txt | grep "Smoothing" > tmp_2.txt
            smoothing=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

            #get convergence factor
            python3 ../error_modes/two_grid.py $E $nu $gamma $v1 $v2 > tmp_1.txt
            cat tmp_1.txt > log_3.txt

            cat tmp_1.txt | grep "Spectral" > tmp_2.txt
            convergence_factor=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')  

            nodes_x=32
            nodes_y=32
            for ((i=0; i<iterations; i++))
            do
                python3 convergence_tester.py $nodes_x $nodes_y $timesteps_mg $coarsest_level_iter $E $nu $v1 $v2 | tee tmp_1.txt
                cat tmp_1.txt > log.txt

                python3 plotter.py $v_tot $smoothing $convergence_factor $E $nu | tee tmp_1.txt
                cat tmp_1.txt > log_4.txt

                cat tmp_1.txt | grep "Actual" > tmp_2.txt
                actual_convergence=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')


                rm tmp*

                dir=nodes_$nodes_x
                mkdir $dir
                mv *.pdf $dir

                echo "============ Iteration $i done =============="
                echo ""

                #coarsest_level_iter=$((coarsest_level_iter*4))
                smoothing_pow=$(echo "$smoothing^$v_tot" | bc -l)
                echo "$nodes_x,$nodes_y,$E,$nu,$v1,$v2,$smoothing_pow,$convergence_factor,$actual_convergence" >> $results_file 

                nodes_x=$((nodes_x*8))
                nodes_y=$((nodes_y*8))

            done
            dir=E_"$E"_nu_"$nu"_v1_"$v1"_v2_"$v2"
            mkdir -p $dir
            rm -rf $dir/nodes*
            mv nodes* $dir
        done
    done
    dir=v1"_"$v1"_v2_"$v2
    mkdir -p $dir
    mv E_* $dir

    v1=$((v1+1))
    v2=$((v2+1))

done