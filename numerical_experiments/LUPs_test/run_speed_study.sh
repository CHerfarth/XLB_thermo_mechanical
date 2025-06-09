#!/bin/bash

nodes_x_base=16
nodes_y_base=16
dt_base=0.01
timesteps=10000

iterations=7
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

repeated_measurements=1

echo "dim,MLUP/s" > $results_file


for ((k=0; k<repeated_measurements; k++))
do
    nodes_x=$nodes_x_base
    nodes_y=$nodes_y_base
    dt=$dt_base
    for ((i=0; i<iterations; i++))
    do
        python3 speed_tester.py $nodes_x $nodes_y $timesteps $dt > tmp_1.txt 
        cat tmp_1.txt >> $log_file #write to log


        cat tmp_1.txt | grep "MLUP/s" > tmp_2.txt
        speed=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

        echo "$nodes_x,$speed" >> $results_file


        nodes_x=$((nodes_x*2))
        nodes_y=$((nodes_y*2))
        dt=$(echo "$dt*0.25"|bc -l)
    done
done

python3 plotter.py $results_file
rm tmp*
