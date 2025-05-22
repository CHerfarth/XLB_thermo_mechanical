#!/bin/bash

nodes_x=128
nodes_y=128
timesteps=200
coarsest_level_iter=100
vals_nu=2
vals_E=10

vals_v=7

current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

echo "nodes_x,nodes_y,timesteps,coarsest_level_iter,nu,E,v1,v2,converged,rate" > $results_file
for ((i=0; i<vals_nu; i++))
do
    nu=$(echo "0.5 + 0.3*$i"|bc -l)

    for ((j=0; j<vals_E; j++))
    do
        E=$(echo "0.1 + $j*0.1"|bc -l)
        
        echo "Testing nu $nu and E $E"
        
        v1=0
        for ((k=0; k<vals_v; k++))
        do
            v2=0
            for ((l=0; l<vals_v; l++))
            do
                python3 convergence_tester.py $nodes_x $nodes_y $timesteps $E $nu $coarsest_level_iter $v1 $v2 | tee tmp_1.txt
                
                echo tmp_1.txt >> $log_file

                #get status of convergence
                cat tmp_1.txt | grep "Converged" > tmp_2.txt
                converged=$(cat tmp_2.txt | grep -oE '[0-9]')

                #get convergence rate
                cat tmp_1.txt | grep "Rate" > tmp_2.txt
                rate=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

                echo "$nodes_x,$nodes_y,$timesteps,$coarsest_level_iter,$nu,$E,$v1,$v2,$converged,$rate" >> $results_file

                v2=$((v2+1))
            done
            v1=$((v1+1))
        done
    done
done

for ((i=0; i<iterations; i++))
do
    python3 convergence_tester.py $nodes_x $nodes_y $timesteps_mg $timesteps_standard $dt $coarsest_level_iter > tmp_1.txt
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
    echo "Expected smoothing factor standard LB $smoothing"

    python3 plotter.py $amplification 4 $smoothing


    rm tmp*

    dir=nodes_$nodes_x
    mkdir $dir
    mv *.png $dir

    echo "============ Iteration $i done =============="
    echo ""

    nodes_x=$((nodes_x*2))
    nodes_y=$((nodes_y*2))
    coarsest_level_iter=$((coarsest_level_iter*4))
    dt=$(echo "$dt*0.25"|bc -l)


done



