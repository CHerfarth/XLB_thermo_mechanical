#!/bin/bash

nodes_x=32
nodes_y=32
timesteps=500
coarsest_level_iter=100
vals_nu=1
vals_E=10

vals_v=7

current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

echo "nodes_x,nodes_y,timesteps,coarsest_level_iter,nu,E,v1,v2,converged,rate,WU_per_iteration" > $results_file
for ((i=0; i<vals_nu; i++))
do
    nu=$(echo "0.8 + 0.1*$i"|bc -l)

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
                
                more tmp_1.txt >> $log_file

                #get status of convergence
                cat tmp_1.txt | grep "Converged" > tmp_2.txt
                converged=$(cat tmp_2.txt | grep -oE '[0-9]')

                #get convergence rate
                cat tmp_1.txt | grep "Rate" > tmp_2.txt
                rate=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

                #get Wu per iteration
                cat tmp_1.txt | grep "WU" > tmp_2.txt
                WU_per_iteration=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')


                echo "$nodes_x,$nodes_y,$timesteps,$coarsest_level_iter,$nu,$E,$v1,$v2,$converged,$rate,$WU_per_iteration" >> $results_file

                rm tmp*

                v2=$((v2+1))
            done
            v1=$((v1+1))
        done
    done
done

python3 plotter.py $results_file



